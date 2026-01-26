"""
GatingBN (Dynamic Channel Gating + BN Anchor) for Deepfake Detection

A test-time adaptation method that uses:
1. Dynamic 1x1 Conv gating (spatial-aware channel reweighting)
2. BN Anchor (source statistics alignment)
3. Loss: L_bn (main) + L_ent (auxiliary)
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from dataclasses import dataclass
from typing import Tuple, Optional
from torch.utils.data import DataLoader


# ============================================================================
# Config
# ============================================================================

@dataclass
class GatingBNConfig:
    """GatingBN Configuration"""
    model: str = "LGrad"                    # "LGrad" or "NPR"
    target_layer: str = None                # Auto-detect

    # Gating parameters
    tau: float = 1.0                        # Temperature for sigmoid
    init_bias: float = 2.0                  # Initial bias (sigmoid(2)≈0.88)

    # Loss weights
    lambda_bn: float = 1.0                  # BN alignment loss weight (main)
    lambda_ent: float = 0.1                 # Entropy loss weight (auxiliary)
    lambda_l1: float = 0.01                 # L1 regularization weight (gate → 1)

    # TTA parameters
    tta_lr: float = 1e-4                    # TTA learning rate
    max_tta_steps: int = 10                 # Max TTA iterations
    enable_tta: bool = True                 # Enable test-time adaptation
    optimizer: str = "Adam"                 # "Adam", "AdamW", or "SGD"
    momentum: float = 0.9                   # Momentum for SGD
    weight_decay: float = 1e-4              # Weight decay for AdamW/SGD

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Components
# ============================================================================

class DynamicChannelGating(nn.Module):
    """
    1x1 Conv based dynamic channel gating.

    Unlike GAP-based methods, this computes different gate values
    for each spatial location, enabling spatially-aware channel selection.
    """

    def __init__(self, channels: int, tau: float = 1.0, init_bias: float = 2.0):
        super().__init__()
        self.channels = channels
        self.tau = tau
        self.init_bias = init_bias

        # 1x1 Conv: learns to predict gate scores per location
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        # Initialize: gate ≈ 1 (preserve original features)
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, self.init_bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Feature map [B, C, H, W]
        Returns:
            x_gated: Gated features [B, C, H, W]
            gate: Gate values [B, C, H, W]
        """
        score = self.conv(x)  # [B, C, H, W]
        gate = torch.sigmoid(score / self.tau)  # [B, C, H, W]
        x_gated = x * gate
        return x_gated, gate


# ============================================================================
# GatingBN Module
# ============================================================================

class GatingBN(nn.Module):
    """
    Dynamic Channel Gating + BN Anchor module.

    Flow:
        F → DynamicGating → F_gated → BN_anchor → F_anchored

    The BN anchor normalizes using source statistics to maintain
    the source manifold, preventing distribution drift from gating.
    """

    def __init__(self, channels: int, config: GatingBNConfig):
        super().__init__()
        self.channels = channels
        self.config = config

        # Dynamic gating
        self.gating = DynamicChannelGating(
            channels,
            tau=config.tau,
            init_bias=config.init_bias
        )

        # Source statistics (to be set via set_source_stats)
        self.register_buffer('mu_src', torch.zeros(channels))
        self.register_buffer('std_src', torch.ones(channels))
        self.stats_initialized = False

    def set_source_stats(self, mu: torch.Tensor, std: torch.Tensor):
        """Set source statistics for BN anchor."""
        self.mu_src.copy_(mu)
        self.std_src.copy_(std)
        self.stats_initialized = True

    def forward(self, F: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            F: Feature map [B, C, H, W]
        Returns:
            F_anchored: BN-anchored features [B, C, H, W]
            F_gated: Gated features (before BN anchor) [B, C, H, W]
            gate: Gate values [B, C, H, W]
        """
        # Dynamic gating
        F_gated, gate = self.gating(F)

        # BN Anchor: normalize with source statistics
        # F_anchored = (F_gated - mu_src) / std_src
        eps = 1e-5
        mu = self.mu_src.view(1, -1, 1, 1)
        std = self.std_src.view(1, -1, 1, 1)
        F_anchored = (F_gated - mu) / (std + eps)

        return F_anchored, F_gated, gate


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


# ============================================================================
# Unified GatingBN Wrapper
# ============================================================================

class UnifiedGatingBN(nn.Module):
    """
    Unified GatingBN wrapper for LGrad and NPR models.

    Key features:
    1. Dynamic 1x1 Conv gating - spatially-aware channel reweighting
    2. BN Anchor - maintains source manifold via statistics alignment
    3. TTA Loss - L_bn (main) + L_ent (auxiliary)

    Args:
        base_model: Pre-trained LGrad or NPR model
        config: GatingBN configuration

    Example:
        >>> from model.LGrad.lgrad_model import LGrad
        >>> from model.method.gating_bn import UnifiedGatingBN, GatingBNConfig
        >>>
        >>> lgrad = LGrad(stylegan_weights="...", classifier_weights="...", device="cuda")
        >>> config = GatingBNConfig(model="LGrad", tta_lr=1e-4, max_tta_steps=10)
        >>> model = UnifiedGatingBN(lgrad, config)
        >>>
        >>> # Collect source statistics first
        >>> model.collect_source_stats(clean_dataloader)
        >>>
        >>> # Inference with TTA
        >>> logits = model(images)
        >>> probs = torch.sigmoid(logits)
    """

    def __init__(self, base_model: nn.Module, config: GatingBNConfig):
        super().__init__()
        self.config = config
        self.base_model = base_model

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

        # GatingBN module
        self.gating_bn = GatingBN(channels, config).to(config.device)

        # Optimizer for continual TTA (initialized lazily)
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

    def _get_layer_channels(self, layer_name: str) -> int:
        dummy_input = torch.zeros(1, 3, 224, 224).to(self.config.device)
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

    def _create_optimizer(self, params):
        """Create optimizer based on config."""
        if self.config.optimizer == "Adam":
            return torch.optim.Adam(params, lr=self.config.tta_lr)
        elif self.config.optimizer == "AdamW":
            return torch.optim.AdamW(
                params,
                lr=self.config.tta_lr,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "SGD":
            return torch.optim.SGD(
                params,
                lr=self.config.tta_lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}. Use 'Adam', 'AdamW', or 'SGD'.")

    def collect_source_stats(self, dataloader: DataLoader):
        """
        Collect source statistics from clean data.

        Args:
            dataloader: DataLoader for clean source data
        """
        all_features = []

        self.base_model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.config.device)
                _ = self.base_model(images)
                feat = self.feature_extractor.features[self.config.target_layer]
                all_features.append(feat)

        all_features = torch.cat(all_features, dim=0)

        # Compute channel-wise statistics
        mu_src = all_features.mean(dim=(0, 2, 3))  # [C]
        std_src = all_features.std(dim=(0, 2, 3), unbiased=False)  # [C]

        # Set to GatingBN module
        self.gating_bn.set_source_stats(mu_src, std_src)

        print(f"Source statistics collected from {len(all_features)} samples")
        print(f"  mu_src: mean={mu_src.mean().item():.4f}, std={mu_src.std().item():.4f}")
        print(f"  std_src: mean={std_src.mean().item():.4f}, std={std_src.std().item():.4f}")

    def set_source_stats(self, mu_src: torch.Tensor, std_src: torch.Tensor):
        """Manually set source statistics."""
        self.gating_bn.set_source_stats(mu_src.to(self.config.device),
                                         std_src.to(self.config.device))

    def _compute_bn_loss(self, F_gated: torch.Tensor) -> torch.Tensor:
        """
        Compute BN alignment loss.

        L_bn = |μ(F_gated) - μ_src| + |σ(F_gated) - σ_src|
        """
        mu_batch = F_gated.mean(dim=(0, 2, 3))  # [C]
        std_batch = F_gated.std(dim=(0, 2, 3), unbiased=False)  # [C]

        loss_mu = (mu_batch - self.gating_bn.mu_src).abs().mean()
        loss_std = (std_batch - self.gating_bn.std_src).abs().mean()

        return loss_mu + loss_std

    def _compute_entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy minimization loss.

        L_ent = -Σ p log p
        """
        eps = 1e-8
        prob = torch.sigmoid(logits)
        entropy = -(prob * (prob + eps).log() + (1 - prob) * (1 - prob + eps).log())
        return entropy.mean()

    def _compute_l1_loss(self, gate: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 regularization loss on gate deviation from 1.

        L_l1 = mean(|1 - gate|)

        This encourages gate values to stay close to 1 (preserve original features),
        preventing entropy minimization from making gates too extreme.
        """
        return (1 - gate).abs().mean()

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

        if not self.gating_bn.stats_initialized:
            raise RuntimeError("Source statistics not initialized. "
                             "Call collect_source_stats() or set_source_stats() first.")

        if enable_tta:
            return self._forward_with_tta(images)
        else:
            return self._forward_no_tta(images)

    def _forward_no_tta(self, images: torch.Tensor) -> torch.Tensor:
        """Forward without TTA."""
        with torch.no_grad():
            _ = self.base_model(images)

        features = self.feature_extractor.features[self.config.target_layer]
        F_anchored, _, _ = self.gating_bn(features)

        # Classifier head
        feat_pooled = self.classifier.avgpool(F_anchored)
        feat_pooled = torch.flatten(feat_pooled, 1)
        logits = self.fc_layer(feat_pooled)

        return logits

    def _forward_with_tta(self, images: torch.Tensor) -> torch.Tensor:
        """Forward with TTA."""
        device = self.config.device
        images = images.to(device)
        eps = 1e-8

        # Get trainable parameters (gating only)
        params_to_update = list(self.gating_bn.gating.parameters())

        if len(params_to_update) == 0:
            return self._forward_no_tta(images)

        # Enable gradients for gating
        for p in self.parameters():
            p.requires_grad = False
        for p in params_to_update:
            p.requires_grad = True

        # Create optimizer only if not exists (continual TTA)
        if self.optimizer is None:
            self.optimizer = self._create_optimizer(params_to_update)

        # TTA loop
        prev_loss = None

        for step in range(self.config.max_tta_steps):
            # Forward through base model (frozen)
            with torch.no_grad():
                _ = self.base_model(images)

            features = self.feature_extractor.features[self.config.target_layer]

            # GatingBN forward (gating is trainable)
            F_anchored, F_gated, gate = self.gating_bn(features)

            # Classifier forward
            feat_pooled = self.classifier.avgpool(F_anchored)
            feat_pooled = torch.flatten(feat_pooled, 1)
            logits = self.fc_layer(feat_pooled)

            # Compute losses
            L_bn = self._compute_bn_loss(F_gated)
            L_ent = self._compute_entropy_loss(logits)
            L_l1 = self._compute_l1_loss(gate)
            loss = (self.config.lambda_bn * L_bn +
                    self.config.lambda_ent * L_ent +
                    self.config.lambda_l1 * L_l1)

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_update, max_norm=1.0)
            self.optimizer.step()

            # Early stopping: loss convergence
            loss_val = loss.item()
            if prev_loss is not None:
                if abs(loss_val - prev_loss) < 1e-4:
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

        Call this when switching to a new domain/dataset to ensure
        fresh adaptation.
        """
        self.gating_bn.gating._init_weights()
        self.optimizer = None  # Will be recreated on next forward

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict real/fake labels."""
        logits = self.forward(x)
        return (torch.sigmoid(logits) > 0.5).long().squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probability of being fake."""
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(1)
