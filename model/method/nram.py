"""
NRAM (Noise-Robust Artifact Modulation) for Deepfake Detection

A test-time adaptation method that enhances features based on artifact and noise levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np


# ============================================================================
# Utilities
# ============================================================================

def robust_normalize(x: torch.Tensor, dim: int = 0, eps: float = None) -> torch.Tensor:
    """Robust normalization using median and MAD."""
    if eps is None:
        eps = torch.finfo(x.dtype).eps * 10

    median = torch.median(x, dim=dim, keepdim=True).values
    mad = torch.median(torch.abs(x - median), dim=dim, keepdim=True).values
    z = (x - median) / (mad + eps)

    return z


def compute_quantile(x: torch.Tensor, q: float, dim: int = 0) -> torch.Tensor:
    """Compute quantile along dimension."""
    k = int(q * x.shape[dim])
    k = max(0, min(k, x.shape[dim] - 1))
    return torch.kthvalue(x, k + 1, dim=dim, keepdim=True).values


# ============================================================================
# Config
# ============================================================================

@dataclass
class NRAMConfig:
    """NRAM Configuration"""
    model: str = "LGrad"                    # "LGrad" or "NPR"
    target_layer: str = None                # Auto-detect
    reduction_ratio: int = 16               # SE reduction
    tta_lr: float = 1e-4                    # TTA learning rate
    max_tta_steps: int = 10                 # Max TTA iterations
    update_sample_ratio: float = 0.7        # Top-ρ sample selection
    enable_tta: bool = True                 # Enable test-time adaptation
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Components
# ============================================================================

class FrequencyArtifactDetector(nn.Module):
    """Detect artifact level using frequency analysis."""

    def __init__(self, num_bands: int = 12):
        super().__init__()
        self.num_bands = num_bands

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """
        Args:
            F: Feature map [B, C, H, W]
        Returns:
            artifact_score: [B, 1] in [0, 1]
        """
        B, C, H, W = F.shape
        device = F.device
        eps = torch.finfo(F.dtype).eps * 10

        # FFT magnitude spectrum
        F_mean = F.mean(dim=1, keepdim=True)
        F_freq = torch.fft.fft2(F_mean.squeeze(1))
        F_mag = torch.abs(F_freq)
        F_mag = torch.fft.fftshift(F_mag, dim=(-2, -1))

        # Radial frequency mask
        y, x = torch.meshgrid(
            torch.arange(H, device=device) - H // 2,
            torch.arange(W, device=device) - W // 2,
            indexing='ij'
        )
        max_radius = np.sqrt((H // 2) ** 2 + (W // 2) ** 2)
        radius = torch.sqrt(x.float() ** 2 + y.float() ** 2) / max_radius

        # Multi-band energy
        band_energies = []
        for i in range(self.num_bands):
            low = i / self.num_bands
            high = (i + 1) / self.num_bands
            mask = ((radius >= low) & (radius < high)).float()
            band_mag = (F_mag * mask.unsqueeze(0)).sum(dim=(-2, -1))
            band_mag_norm = band_mag / (mask.sum() + eps)
            band_energies.append(band_mag_norm)

        e = torch.stack(band_energies, dim=1)  # [B, num_bands]
        e_z = robust_normalize(e, dim=1)
        artifact_z = e_z.max(dim=1, keepdim=True).values
        artifact_score = torch.sigmoid(artifact_z)

        return artifact_score


class NoiseEstimator(nn.Module):
    """Estimate noise level using Laplacian filter."""

    def __init__(self):
        super().__init__()
        laplacian_kernel = torch.tensor([
            [0., -1., 0.],
            [-1., 4., -1.],
            [0., -1., 0.]
        ]).view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel / laplacian_kernel.abs().sum()
        self.register_buffer('laplacian_kernel', laplacian_kernel)

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """
        Args:
            F: Feature map [B, C, H, W]
        Returns:
            noise_level: [B, 1] in [0, 1]
        """
        B, C, H, W = F.shape
        kernel = self.laplacian_kernel.to(F.device)

        F_reshaped = F.view(B * C, 1, H, W)
        F_filtered = torch.nn.functional.conv2d(F_reshaped, kernel, padding=1)
        F_filtered = F_filtered.view(B, C, H, W)

        spatial_var = F_filtered.var(dim=[2, 3])
        noise_var = spatial_var.mean(dim=1, keepdim=True)
        noise_z = robust_normalize(noise_var, dim=0)
        noise_level = torch.sigmoid(noise_z)

        return noise_level


class ChannelAttention(nn.Module):
    """SE-style channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.attention(F))


class ArtifactConditionalChannelAttention(nn.Module):
    """Dual-path channel attention conditioned on artifact level."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.low_artifact_attn = ChannelAttention(channels, reduction)
        self.high_artifact_attn = ChannelAttention(channels, reduction)

    def forward(self, F: torch.Tensor, artifact_score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            F: [B, C, H, W]
            artifact_score: [B, 1]
        Returns:
            attn: [B, C, 1, 1]
        """
        attn_low = self.low_artifact_attn(F)
        attn_high = self.high_artifact_attn(F)
        artifact_4d = artifact_score.view(-1, 1, 1, 1)
        attn = (1 - artifact_4d) * attn_low + artifact_4d * attn_high
        return attn


# ============================================================================
# NRAM Module
# ============================================================================

class NRAM(nn.Module):
    """
    Noise-Robust Artifact Modulation layer.

    Enhances features using artifact-conditional attention and noise gating.
    """

    def __init__(self, channels: int, config: NRAMConfig):
        super().__init__()
        self.channels = channels
        self.config = config

        self.artifact_detector = FrequencyArtifactDetector()
        self.noise_estimator = NoiseEstimator()
        self.artifact_attention = ArtifactConditionalChannelAttention(
            channels, config.reduction_ratio
        )
        self.register_buffer('alpha', torch.tensor(0.1))  # Fixed residual weight

    def forward(self, F: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            F: Feature map [B, C, H, W]
        Returns:
            F_enhanced: Enhanced features [B, C, H, W]
            artifact_score: [B, 1]
            noise_level: [B, 1]
        """
        B, C, H, W = F.shape

        # Detect artifact and noise (no grad for evidence)
        with torch.no_grad():
            artifact_score = self.artifact_detector(F)
            noise_level = self.noise_estimator(F)

        # Artifact-conditional attention
        attn = self.artifact_attention(F, artifact_score.detach())

        # Noise-based gating
        gate = 1.0 - noise_level.detach()
        gate = gate.view(B, 1, 1, 1).expand(-1, C, 1, 1)

        # Combine attention and gating
        weights = attn * gate
        F_gated = F * weights

        # Residual connection
        F_enhanced = (1 - self.alpha) * F_gated + self.alpha * F

        return F_enhanced, artifact_score, noise_level


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
# Unified NRAM Wrapper
# ============================================================================

class UnifiedNRAM(nn.Module):
    """
    Unified NRAM wrapper for LGrad and NPR models.

    Args:
        base_model: Pre-trained LGrad or NPR model
        config: NRAM configuration

    Example:
        >>> from model.LGrad.lgrad_model import LGrad
        >>> from model.method.nram import UnifiedNRAM, NRAMConfig
        >>>
        >>> lgrad = LGrad(stylegan_weights="...", classifier_weights="...", device="cuda")
        >>> config = NRAMConfig(model="LGrad", tta_lr=1e-4, max_tta_steps=10)
        >>> model = UnifiedNRAM(lgrad, config)
        >>>
        >>> # Inference with TTA
        >>> logits = model(images)
        >>> probs = torch.sigmoid(logits)
    """

    def __init__(self, base_model: nn.Module, config: NRAMConfig):
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

        # NRAM module
        self.nram = NRAM(channels, config).to(config.device)

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
        with torch.no_grad():
            _ = self.base_model(images)

        features = self.feature_extractor.features[self.config.target_layer]
        feat_enhanced, _, _ = self.nram(features)

        # Use base model's classifier
        if self.config.model == "LGrad":
            classifier = self.base_model.classifier
        else:
            classifier = self.base_model.model

        feat_pooled = classifier.avgpool(feat_enhanced)
        feat_pooled = torch.flatten(feat_pooled, 1)
        fc_layer = classifier.fc1 if self.config.model == "NPR" else classifier.fc
        logits = fc_layer(feat_pooled)

        return logits

    def _forward_with_tta(self, images: torch.Tensor) -> torch.Tensor:
        """Forward with TTA."""
        device = self.config.device
        images = images.to(device)
        B = images.shape[0]
        eps = torch.finfo(torch.float32).eps * 10

        # Initial forward
        with torch.no_grad():
            logits_init = self._forward_no_tta(images)

        # Get TTA parameters
        params_to_update = [p for n, p in self.named_parameters()
                           if 'artifact_attention' in n]

        if len(params_to_update) == 0:
            return logits_init

        # Enable gradients
        for p in self.parameters():
            p.requires_grad = False
        for p in params_to_update:
            p.requires_grad = True

        optimizer = torch.optim.Adam(params_to_update, lr=self.config.tta_lr)

        # TTA loop with early stopping
        prev_loss = None
        prev_q_mean = 0.0
        collapse_counter = 0

        for step in range(self.config.max_tta_steps):
            # Forward
            with torch.no_grad():
                _ = self.base_model(images)

            features = self.feature_extractor.features[self.config.target_layer]
            feat_enhanced, artifact_score, noise_level = self.nram(features)

            # Classifier
            if self.config.model == "LGrad":
                classifier = self.base_model.classifier
            else:
                classifier = self.base_model.model

            feat_pooled = classifier.avgpool(feat_enhanced)
            feat_pooled = torch.flatten(feat_pooled, 1)
            fc_layer = classifier.fc1 if self.config.model == "NPR" else classifier.fc
            logits = fc_layer(feat_pooled)
            prob = torch.sigmoid(logits)

            # Evidence quality
            with torch.no_grad():
                q = (artifact_score - noise_level + 1.0) / 2.0
                q = torch.clamp(q, 0.0, 1.0).squeeze(1)
                q_mean_step = q.mean().item()

                # Sample selection (top-ρ)
                q_threshold = compute_quantile(q, 1.0 - self.config.update_sample_ratio, dim=0).item()
                mask = (q >= q_threshold).float().unsqueeze(1)

                if mask.sum() == 0:
                    mask = torch.ones_like(mask)

                # Dynamic LR scaling based on evidence quality
                gating_factor = torch.sigmoid(torch.tensor(2.0 * (q_mean_step - 0.5))).item()
                effective_lr = self.config.tta_lr * gating_factor

                for param_group in optimizer.param_groups:
                    param_group['lr'] = effective_lr

            # Entropy loss on selected samples
            entropy = -(prob * torch.log(prob + eps) +
                       (1 - prob) * torch.log(1 - prob + eps))
            loss = (mask * entropy).sum() / (mask.sum() + eps)

            # Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_update, max_norm=1.0)
            optimizer.step()

            # Early stopping checks
            prob_mean = prob.mean().item()
            prob_std = prob.std().item()
            loss_val = loss.item()

            # Check 1: Loss convergence
            converged = False
            if prev_loss is not None:
                loss_change = abs(loss_val - prev_loss)
                relative_eps = 0.01 * abs(prev_loss) if abs(prev_loss) > 1e-6 else 1e-4
                if loss_change < relative_eps:
                    converged = True

            # Check 2: Evidence stability
            evidence_stable = False
            q_change = abs(q_mean_step - prev_q_mean)
            if q_change < 0.05:
                evidence_stable = True

            # Check 3: Collapse detection
            collapse_detected = False
            if prob_mean < 0.01 or prob_mean > 0.99:
                collapse_detected = True
            if prob_std < 0.05:
                collapse_detected = True

            if collapse_detected:
                collapse_counter += 1
                if collapse_counter >= 2:
                    break
            else:
                collapse_counter = 0

            # Early stop if converged AND evidence stable
            if converged and evidence_stable:
                break

            prev_loss = loss_val
            prev_q_mean = q_mean_step

        # Final forward
        with torch.no_grad():
            logits_final = self._forward_no_tta(images)

        # Disable gradients
        for p in self.parameters():
            p.requires_grad = False

        return logits_final

    def reset(self):
        """
        Reset NRAM module to initial state.

        Call this when switching to a new dataset to ensure
        the model doesn't carry over adaptation from previous data.
        """
        # Reinitialize NRAM module parameters
        for module in self.nram.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict real/fake labels."""
        logits = self.forward(x)
        return (torch.sigmoid(logits) > 0.5).long().squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probability of being fake."""
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(1)

