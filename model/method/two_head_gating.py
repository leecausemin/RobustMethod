"""
Two-Head Gating + BN Anchor for Deepfake Detection

A test-time adaptation method that uses:
1. Two-head channel gating (artifact-preserve + nuisance-control)
2. BN Anchor (backbone's BN statistics for normalization)
3. Loss: L_ent + L_a + L_n + L_over (orthogonal control)

Key insight:
- EM alone has confirmation bias and model collapse risk
- Two-head design with non-overlap constraint prevents "easy collapse solution"
- Artifact head preserves important features, nuisance head suppresses noise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


# ============================================================================
# Config
# ============================================================================

@dataclass
class TwoHeadGatingConfig:
    """Two-Head Gating Configuration"""
    model: str = "LGrad"                    # "LGrad" or "NPR"
    target_layer: str = None                # Auto-detect

    # Two-head gating parameters
    tau_a: float = 1.0                      # Artifact head temperature
    tau_n: float = 1.0                      # Nuisance head temperature
    init_bias_a: float = 2.0                # g_a ≈ 0.88 (preserve)
    init_bias_n: float = 2.0                # g_n ≈ 0.88 (neutral start)
    gamma: float = 0.25                     # Nuisance control strength

    # Loss weights
    lambda_ent: float = 0.1                 # Entropy minimization
    lambda_a: float = 0.1                   # Artifact preservation
    lambda_n: float = 0.05                  # Nuisance budget constraint
    lambda_over: float = 0.1                # Overlap penalty
    rho: float = 0.1                        # Nuisance budget (allowed suppression)

    # Optimizer (separate lr for each head)
    lr_a: float = 1e-5                      # Artifact head (slow)
    lr_n: float = 1e-4                      # Nuisance head (fast)
    optimizer: str = "Adam"                 # "Adam", "AdamW", or "SGD"
    momentum: float = 0.9                   # Momentum for SGD
    weight_decay: float = 1e-4              # Weight decay for AdamW/SGD

    # TTA parameters
    max_tta_steps: int = 5                  # Reduced from 10 to prevent collapse
    enable_tta: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Components
# ============================================================================

class TwoHeadChannelGating(nn.Module):
    """
    Two-head channel gating module.

    - conv_a: Artifact-preserve head (maintains important features)
    - conv_n: Nuisance-control head (suppresses noise when needed)

    Gate computation:
        score_a, score_n = conv_a(F), conv_n(F)
        g_a = sigmoid(GAP(score_a) / tau_a)
        g_n = sigmoid(GAP(score_n) / tau_n)
        g = clip(g_a + γ(g_n - 1), 0, 1)

    Intuition:
        - g_a: base pass-through rate (should stay near 1)
        - g_n: adjustment amount (only moves down from 1 when needed)
        - γ controls how much nuisance head can affect the final gate
    """

    def __init__(self, channels: int, config: TwoHeadGatingConfig):
        super().__init__()
        self.channels = channels
        self.config = config

        # Two heads: 1x1 conv for channel-wise scoring
        self.conv_a = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.conv_n = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable starting point."""
        # Artifact head: preserve (g_a ≈ 0.88)
        nn.init.zeros_(self.conv_a.weight)
        nn.init.constant_(self.conv_a.bias, self.config.init_bias_a)

        # Nuisance head: neutral start (g_n ≈ 0.88)
        nn.init.zeros_(self.conv_n.weight)
        nn.init.constant_(self.conv_n.bias, self.config.init_bias_n)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Feature map [B, C, H, W]

        Returns:
            x_gated: Gated features [B, C, H, W]
            g: Final gate [B, C, 1, 1]
            g_a: Artifact gate [B, C, 1, 1]
            g_n: Nuisance gate [B, C, 1, 1]
        """
        # 1x1 Conv for each head
        score_a = self.conv_a(x)  # [B, C, H, W]
        score_n = self.conv_n(x)  # [B, C, H, W]

        # GAP for channel-wise gate
        score_a = score_a.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        score_n = score_n.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

        # Sigmoid with temperature
        g_a = torch.sigmoid(score_a / self.config.tau_a)
        g_n = torch.sigmoid(score_n / self.config.tau_n)

        # Residual-style composition: g = clip(g_a + γ(g_n - 1), 0, 1)
        # - g_a is base pass-through
        # - γ(g_n - 1) is adjustment (negative when g_n < 1)
        g = torch.clamp(g_a + self.config.gamma * (g_n - 1), 0, 1)

        # Apply gate (broadcast [B,C,1,1] to [B,C,H,W])
        x_gated = x * g

        return x_gated, g, g_a, g_n


# ============================================================================
# Two-Head Gating Module with BN Anchor
# ============================================================================

class TwoHeadGatingBN(nn.Module):
    """
    Two-Head Channel Gating + BN Anchor module.

    Flow:
        F → TwoHeadGating → F_gated → BN_anchor → F_anchored

    The BN anchor normalizes using backbone's BN statistics (running_mean, running_var).
    """

    def __init__(self, channels: int, config: TwoHeadGatingConfig):
        super().__init__()
        self.channels = channels
        self.config = config

        # Two-head gating
        self.gating = TwoHeadChannelGating(channels, config)

        # BN layer copied from backbone
        self.bn_anchor = nn.BatchNorm2d(channels, affine=False)
        self.bn_initialized = False

    def set_bn_from_backbone(self, bn_layer: nn.BatchNorm2d):
        """Copy running statistics from backbone's BN layer."""
        with torch.no_grad():
            self.bn_anchor.running_mean.copy_(bn_layer.running_mean)
            self.bn_anchor.running_var.copy_(bn_layer.running_var)
        self.bn_anchor.eval()  # Always use running stats
        self.bn_initialized = True

    def forward(self, F: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            F: Feature map [B, C, H, W]

        Returns:
            F_anchored: BN-anchored features [B, C, H, W]
            F_gated: Gated features (before BN) [B, C, H, W]
            g: Final gate [B, C, 1, 1]
            g_a: Artifact gate [B, C, 1, 1]
            g_n: Nuisance gate [B, C, 1, 1]
        """
        # Two-head gating
        F_gated, g, g_a, g_n = self.gating(F)

        # BN Anchor
        F_anchored = self.bn_anchor(F_gated)

        return F_anchored, F_gated, g, g_a, g_n


# ============================================================================
# Feature Extractor
# ============================================================================

class FeatureExtractor:
    """Hook-based feature extractor."""

    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.features = {}
        self.hooks = []

        for name, module in model.named_modules():
            if name == target_layer:
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)
                break

    def _hook_fn(self, name: str):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        self.remove_hooks()


# ============================================================================
# Unified Two-Head Gating Wrapper
# ============================================================================

class UnifiedTwoHeadGating(nn.Module):
    """
    Unified Two-Head Gating wrapper for LGrad and NPR models.

    Key features:
    1. Two-head channel gating - artifact-preserve + nuisance-control
    2. BN Anchor - uses backbone's BN statistics
    3. Orthogonal control - prevents both heads from suppressing same channels
    4. TTA Loss: L_ent + L_a + L_n + L_over

    Novelty over existing methods:
    - T²A uses negative learning/gradient masking to prevent EM collapse
    - We decompose the gate module into two roles with non-overlap constraint,
      structurally removing the "collapsible solution space"

    Args:
        base_model: Pre-trained LGrad or NPR model
        config: TwoHeadGatingConfig

    Example:
        >>> from model.LGrad.lgrad_model import LGrad
        >>> from model.method.two_head_gating import UnifiedTwoHeadGating, TwoHeadGatingConfig
        >>>
        >>> lgrad = LGrad(stylegan_weights="...", classifier_weights="...", device="cuda")
        >>> config = TwoHeadGatingConfig(model="LGrad")
        >>> model = UnifiedTwoHeadGating(lgrad, config)
        >>>
        >>> # Inference with TTA
        >>> logits = model(images)
    """

    def __init__(self, base_model: nn.Module, config: TwoHeadGatingConfig):
        super().__init__()
        self.config = config
        self.base_model = base_model

        # Detect base_model's device
        self.device = next(base_model.parameters()).device

        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False

        # Auto-detect target layer
        if config.target_layer is None:
            config.target_layer = self._get_default_target_layer()

        # Feature extractor
        self.feature_extractor = FeatureExtractor(base_model, config.target_layer)

        # Get channel count
        channels = self._get_layer_channels(config.target_layer)

        # Two-Head Gating + BN module (use detected device)
        self.gating_bn = TwoHeadGatingBN(channels, config).to(self.device)

        # Copy BN statistics from backbone
        self._init_bn_from_backbone()

        # Optimizer (initialized lazily)
        self.optimizer = None

        # Cache classifier reference
        self._init_classifier_ref()

    def _get_default_target_layer(self) -> str:
        if self.config.model == "LGrad":
            return 'classifier.layer4'
        elif self.config.model == "NPR":
            return 'model.layer2'
        else:
            raise ValueError(f"Unknown model: {self.config.model}")

    def _init_bn_from_backbone(self):
        """Find and copy BN layer from the last block of target layer."""
        target_layer = self.config.target_layer

        if self.config.model == "LGrad":
            layer = self.base_model.classifier.layer4
        else:  # NPR
            layer = self.base_model.model.layer2

        # Get the last block
        last_block = layer[-1]

        # Find the last BN (bn3 for Bottleneck, bn2 for BasicBlock)
        if hasattr(last_block, 'bn3'):
            bn_layer = last_block.bn3
        elif hasattr(last_block, 'bn2'):
            bn_layer = last_block.bn2
        else:
            raise RuntimeError(f"Cannot find BN layer in {last_block}")

        self.gating_bn.set_bn_from_backbone(bn_layer)
        print(f"BN Anchor initialized from {target_layer}[-1].bn{'3' if hasattr(last_block, 'bn3') else '2'}")

    def _get_layer_channels(self, layer_name: str) -> int:
        dummy_input = torch.zeros(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            _ = self.base_model(dummy_input)
        feature = self.feature_extractor.features[layer_name]
        return feature.shape[1]

    def _init_classifier_ref(self):
        """Cache classifier and fc layer references."""
        if self.config.model == "LGrad":
            self.classifier = self.base_model.classifier
            self.fc_layer = self.classifier.fc
        else:  # NPR
            self.classifier = self.base_model.model
            self.fc_layer = self.classifier.fc1

    def _create_optimizer(self):
        """Create optimizer with separate parameter groups for each head."""
        param_groups = [
            {
                'params': self.gating_bn.gating.conv_a.parameters(),
                'lr': self.config.lr_a,
                'name': 'artifact_head'
            },
            {
                'params': self.gating_bn.gating.conv_n.parameters(),
                'lr': self.config.lr_n,
                'name': 'nuisance_head'
            },
        ]

        if self.config.optimizer == "Adam":
            return torch.optim.Adam(param_groups)
        elif self.config.optimizer == "AdamW":
            return torch.optim.AdamW(
                param_groups,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "SGD":
            return torch.optim.SGD(
                param_groups,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}. Use 'Adam', 'AdamW', or 'SGD'.")

    # ========================================================================
    # Loss Functions
    # ========================================================================

    def _compute_entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        L_ent: Entropy minimization loss.

        Encourages confident predictions (low entropy).
        """
        eps = 1e-8
        prob = torch.sigmoid(logits)
        entropy = -(prob * (prob + eps).log() + (1 - prob) * (1 - prob + eps).log())
        return entropy.mean()

    def _compute_artifact_loss(self, g_a: torch.Tensor) -> torch.Tensor:
        """
        L_a = ||1 - g_a||_1

        Artifact head should stay near 1 (preserve features).
        Stronger constraint than the original L1 loss on final gate.
        """
        return (1 - g_a).abs().mean()

    def _compute_nuisance_loss(self, g_n: torch.Tensor) -> torch.Tensor:
        """
        L_n = ReLU(mean(1 - g_n) - ρ)²

        Budget constraint: nuisance head can suppress up to ρ on average.
        Beyond that, penalty increases quadratically.
        """
        suppression = (1 - g_n).mean()
        return F.relu(suppression - self.config.rho) ** 2

    def _compute_overlap_loss(self, g_a: torch.Tensor, g_n: torch.Tensor) -> torch.Tensor:
        """
        L_over = mean((1 - g_a) ⊙ (1 - g_n))

        Prevents both heads from suppressing the same channel.
        If a channel needs suppression, only ONE head should do it.
        This is the key constraint preventing EM collapse.
        """
        return ((1 - g_a) * (1 - g_n)).mean()

    # ========================================================================
    # Forward Methods
    # ========================================================================

    def forward(self, images: torch.Tensor, enable_tta: bool = None) -> torch.Tensor:
        """
        Forward pass with optional TTA.

        Args:
            images: [B, 3, H, W]
            enable_tta: Override config.enable_tta if provided

        Returns:
            logits: [B, 1]
        """
        if enable_tta is None:
            enable_tta = self.config.enable_tta

        if enable_tta:
            return self._forward_with_tta(images)
        else:
            return self._forward_no_tta(images)

    def _forward_no_tta(self, images: torch.Tensor) -> torch.Tensor:
        """Forward without TTA."""
        images = images.to(self.device)
        with torch.no_grad():
            _ = self.base_model(images)

        features = self.feature_extractor.features[self.config.target_layer]
        F_anchored, _, _, _, _ = self.gating_bn(features)

        # Classifier head
        feat_pooled = self.classifier.avgpool(F_anchored)
        feat_pooled = torch.flatten(feat_pooled, 1)
        logits = self.fc_layer(feat_pooled)

        return logits

    def _forward_with_tta(self, images: torch.Tensor) -> torch.Tensor:
        """Forward with TTA using two-head gating and orthogonal control."""
        images = images.to(self.device)

        # Get trainable parameters (both heads)
        params_to_update = list(self.gating_bn.gating.parameters())

        if len(params_to_update) == 0:
            return self._forward_no_tta(images)

        # Enable gradients for gating
        for p in self.parameters():
            p.requires_grad = False
        for p in params_to_update:
            p.requires_grad = True

        # Create optimizer if not exists
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()

        # TTA loop
        prev_loss = None

        for step in range(self.config.max_tta_steps):
            # Forward through base model (frozen)
            with torch.no_grad():
                _ = self.base_model(images)

            features = self.feature_extractor.features[self.config.target_layer]

            # Two-head gating forward
            F_anchored, F_gated, g, g_a, g_n = self.gating_bn(features)

            # Classifier forward
            feat_pooled = self.classifier.avgpool(F_anchored)
            feat_pooled = torch.flatten(feat_pooled, 1)
            logits = self.fc_layer(feat_pooled)

            # === Compute losses ===
            L_ent = self._compute_entropy_loss(logits)
            L_a = self._compute_artifact_loss(g_a)
            L_n = self._compute_nuisance_loss(g_n)
            L_over = self._compute_overlap_loss(g_a, g_n)

            loss = (self.config.lambda_ent * L_ent +
                    self.config.lambda_a * L_a +
                    self.config.lambda_n * L_n +
                    self.config.lambda_over * L_over)

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_update, max_norm=1.0)
            self.optimizer.step()

            # Early stopping: loss convergence
            loss_val = loss.item()
            if prev_loss is not None:
                if abs(loss_val - prev_loss) < 1e-5:
                    break
            prev_loss = loss_val

        # Final forward
        with torch.no_grad():
            logits_final = self._forward_no_tta(images)

        # Disable gradients
        for p in self.parameters():
            p.requires_grad = False

        return logits_final

    def reset(self):
        """
        Reset gating module and optimizer to initial state.

        Call this when switching to a new domain/dataset.
        """
        self.gating_bn.gating._init_weights()
        self.optimizer = None

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict real/fake labels."""
        logits = self.forward(x)
        return (torch.sigmoid(logits) > 0.5).long().squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probability of being fake."""
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(1)

    def get_gate_stats(self, images: torch.Tensor) -> dict:
        """
        Get gate statistics for analysis.

        Returns dict with g, g_a, g_n statistics.
        """
        images = images.to(self.device)
        with torch.no_grad():
            _ = self.base_model(images)
            features = self.feature_extractor.features[self.config.target_layer]
            _, _, g, g_a, g_n = self.gating_bn(features)

        return {
            'g_mean': g.mean().item(),
            'g_std': g.std().item(),
            'g_a_mean': g_a.mean().item(),
            'g_a_std': g_a.std().item(),
            'g_n_mean': g_n.mean().item(),
            'g_n_std': g_n.std().item(),
            'overlap': ((1 - g_a) * (1 - g_n)).mean().item(),
        }
