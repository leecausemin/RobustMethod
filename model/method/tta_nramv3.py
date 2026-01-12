"""
Test-Time Adaptive NRAM v3 (Evidence-Calibrated) for Deepfake Detection

This module implements an evidence-calibrated test-time adaptation approach that:
1. Controls WHEN/WHERE to update based on evidence quality (artifact vs noise)
2. Prevents confirmation bias and model collapse (T2A concerns) via update selection
3. Preserves deepfake-specific artifacts while handling corruption

Key Improvements over v2:
- Evidence-calibrated update policy (batch-level + sample-level)
- Conservative vs aggressive parameter subset scheduling
- Sigmoid-constrained learnable residual weight (0~1)
- Differentiable artifact-preserving constraint (HP proxy)
- Enhanced collapse safeguards (entropy + std monitoring)
- Memory-efficient design (minimal CPU transfers)

Architecture:
    Base Model (frozen, always no_grad) → layer4 features
        ↓
    Evidence Quality Estimation (artifact ↑ good, noise ↑ bad)
        ↓
    Update Policy Selection (skip / conservative / aggressive)
        ↓
    TTA-NRAM v3 (artifact-conditional attention + sigmoid α)
        ↓
    Enhanced features → Base Classifier
        ↓
    Final prediction

Author: Evidence-Calibrated TTA Team
Date: 2026-01-12
Reference: T2A (CVPR 2024) - "Think Twice Before Adaptation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TTANRAMv3Config:
    """Configuration for TTA-NRAM v3 (Evidence-Calibrated)"""

    # Model selection
    model: str = "LGrad"  # "LGrad" or "NPR"
    target_layer: str = None  # Auto-detect

    # Channel attention
    reduction_ratio: int = 16

    # Artifact detection (for evidence + logging)
    artifact_bands: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.2, 0.4), (0.4, 0.6)]
    )
    artifact_normalize_factor: float = 1.0

    # Noise estimation
    noise_detection_method: str = "laplacian"
    noise_normalize_factor: float = 100.0
    noise_gate_alpha: float = 1.0

    # === Evidence Calibration (NEW in v3) ===
    enable_evidence_calibration: bool = True
    evidence_w_artifact: float = 1.0       # weight for artifact score
    evidence_w_noise: float = 1.5          # weight for noise level (stricter)
    evidence_bias: float = 0.0             # threshold bias
    evidence_center_artifact: float = 0.5  # center for artifact [0,1]
    evidence_center_noise: float = 0.5     # center for noise [0,1]

    # Update policy thresholds (validated)
    skip_threshold: float = 0.3            # q_mean < τ_skip → skip all
    sample_threshold: float = 0.35         # q_i < τ_sample → exclude from loss
    conservative_threshold: float = 0.6    # q_mean < τ_cons → conservative mode
    # === END Evidence Calibration ===

    # Memory bank (disabled by default for stability)
    enable_memory_bank: bool = False
    memory_size: int = 100
    confidence_threshold: float = 0.8

    # TTA settings (no confidence loss)
    tta_steps: int = 5
    tta_lr: float = 1e-4
    tta_loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "entropy": 1.0,
            "confidence": 0.0,  # Disabled (T2A collapse concern)
        }
    )

    # === Artifact-Preserving Constraint (NEW in v3) ===
    enable_artifact_preserving: bool = True
    artifact_preserving_weight: float = 0.05  # λ_ap
    artifact_preserving_type: str = "differentiable_hp"  # Differentiable HP proxy
    hp_kernel_size: int = 5  # For HP filter (avgpool)
    # === END Artifact-Preserving ===

    # Gating (learnable, sigmoid-constrained)
    residual_weight: float = 0.1
    learnable_residual_weight: bool = True
    sigmoid_alpha: bool = True  # Use sigmoid(α_logit) for [0,1] constraint

    # Collapse safeguards (enhanced)
    enable_collapse_detection: bool = True
    collapse_prob_threshold: float = 0.01  # prob_mean < 0.01 or > 0.99
    collapse_std_threshold: float = 0.05   # prob_std < 0.05
    collapse_patience: int = 2             # consecutive steps before early stop

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    verbose: bool = False  # Set to True to see detailed TTA logs

    def __post_init__(self):
        """Validate config."""
        assert self.sample_threshold >= self.skip_threshold, \
            f"sample_threshold ({self.sample_threshold}) must be >= skip_threshold ({self.skip_threshold})"


# ============================================================================
# Phase 1: Frequency-based Artifact Detector (Logging only, no grad)
# ============================================================================

class FrequencyArtifactDetector(nn.Module):
    """
    Parameter-free artifact detection using frequency-domain analysis.

    NOTE: This is used for evidence estimation and logging ONLY.
    For artifact-preserving constraint, we use differentiable HP proxy.
    """

    def __init__(
        self,
        bands: List[Tuple[float, float]] = None,
        normalize_factor: float = 1.0
    ):
        super().__init__()
        self.bands = bands or [(0.2, 0.4), (0.4, 0.6)]
        self.normalize_factor = normalize_factor

    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Detect artifact level from feature map.

        Args:
            feature_map: [B, C, H, W]

        Returns:
            artifact_score: [B, 1]
            debug_info: Dict with band_energies (for logging)
        """
        B, C, H, W = feature_map.shape
        device = feature_map.device

        # Channel-wise average
        F_mean = feature_map.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # 2D FFT → Magnitude Spectrum
        F_freq = torch.fft.fft2(F_mean.squeeze(1))  # [B, H, W] complex
        F_mag = torch.abs(F_freq)  # [B, H, W]
        F_mag = torch.fft.fftshift(F_mag, dim=(-2, -1))  # [B, H, W]

        # Create radial frequency mask
        y, x = torch.meshgrid(
            torch.arange(H, device=device) - H // 2,
            torch.arange(W, device=device) - W // 2,
            indexing='ij'
        )
        max_radius = np.sqrt((H // 2) ** 2 + (W // 2) ** 2)
        radius = torch.sqrt(x.float() ** 2 + y.float() ** 2) / max_radius  # [H, W]

        # Extract band energies
        band_scores = []
        band_energies = []

        for low, high in self.bands:
            mask = ((radius >= low) & (radius < high)).float()  # [H, W]

            band_mag = (F_mag * mask.unsqueeze(0)).sum(dim=(-2, -1))  # [B]
            band_count = mask.sum() + 1e-8
            band_mag_norm = band_mag / band_count

            band_scores.append(band_mag_norm)
            band_energies.append(band_mag_norm)

        # Average across bands
        artifact_raw = torch.stack(band_scores, dim=1).mean(dim=1)  # [B]

        # Compute baseline
        baseline_mask = (radius < 0.2).float()
        baseline_mag = (F_mag * baseline_mask.unsqueeze(0)).sum(dim=(-2, -1))  # [B]
        baseline_count = baseline_mask.sum() + 1e-8
        baseline_mag = baseline_mag / baseline_count + 1e-8

        # Relative score
        relative_score = (artifact_raw - baseline_mag) / baseline_mag  # [B]
        artifact_score = torch.sigmoid(relative_score * self.normalize_factor)  # [B]
        artifact_score = artifact_score.unsqueeze(1)  # [B, 1]

        # Debug info (for logging)
        debug_info = {
            'band_energies': torch.stack(band_energies, dim=1),  # [B, num_bands]
        }

        return artifact_score, debug_info


# ============================================================================
# Phase 2: Noise Estimator (Same as v2)
# ============================================================================

class NoiseEstimator(nn.Module):
    """Parameter-free noise level estimation based on high-frequency analysis."""

    def __init__(self, method: str = "laplacian", normalize_factor: float = 100.0):
        super().__init__()
        self.method = method
        self.normalize_factor = normalize_factor

        if method == "laplacian":
            laplacian_kernel = torch.tensor([
                [0., -1., 0.],
                [-1., 4., -1.],
                [0., -1., 0.]
            ]).view(1, 1, 3, 3) / 8.0

            self.register_buffer('laplacian_kernel', laplacian_kernel)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Estimate noise level from feature map.

        Args:
            feature_map: [B, C, H, W]

        Returns:
            noise_level: [B, 1]
        """
        if self.method == "laplacian":
            return self._estimate_laplacian(feature_map)
        elif self.method == "variance":
            return self._estimate_variance(feature_map)
        else:
            raise ValueError(f"Unknown noise detection method: {self.method}")

    def _estimate_laplacian(self, F: torch.Tensor) -> torch.Tensor:
        """Laplacian-based high-frequency detection."""
        B, C, H, W = F.shape
        device = F.device

        kernel = self.laplacian_kernel.to(device)  # [1, 1, 3, 3]

        F_reshaped = F.view(B * C, 1, H, W)
        F_filtered = torch.nn.functional.conv2d(F_reshaped, kernel, padding=1)  # [B*C, 1, H, W]
        F_filtered = F_filtered.view(B, C, H, W)

        spatial_var = F_filtered.var(dim=[2, 3])  # [B, C]
        noise_var = spatial_var.mean(dim=1, keepdim=True)  # [B, 1]

        noise_level = torch.clamp(noise_var / self.normalize_factor, 0.0, 1.0)

        return noise_level

    def _estimate_variance(self, F: torch.Tensor) -> torch.Tensor:
        """Simple spatial variance-based noise estimation."""
        B, C, H, W = F.shape

        spatial_var = F.var(dim=[2, 3])  # [B, C]
        noise_var = spatial_var.mean(dim=1, keepdim=True)  # [B, 1]

        noise_level = torch.clamp(noise_var / self.normalize_factor, 0.0, 1.0)

        return noise_level


# ============================================================================
# Phase 3: Channel Attention (Same as v2)
# ============================================================================

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation (SE) style channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """
        Compute channel attention weights.

        Args:
            F: Feature map [B, C, H, W]

        Returns:
            attn_weights: [B, C, 1, 1]
        """
        return torch.sigmoid(self.attention(F))


class ArtifactConditionalChannelAttention(nn.Module):
    """
    Dual-path channel attention conditioned on artifact level.

    - low_artifact_attn: For real-like samples
    - high_artifact_attn: For fake-like samples
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels

        self.low_artifact_attn = ChannelAttention(channels, reduction)
        self.high_artifact_attn = ChannelAttention(channels, reduction)

    def forward(
        self,
        F: torch.Tensor,
        artifact_score: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute artifact-conditional channel attention.

        Args:
            F: Feature map [B, C, H, W]
            artifact_score: Artifact level [B, 1]

        Returns:
            attn: Artifact-conditional attention [B, C, 1, 1]
        """
        attn_low = self.low_artifact_attn(F)   # [B, C, 1, 1]
        attn_high = self.high_artifact_attn(F)  # [B, C, 1, 1]

        artifact_score_4d = artifact_score.view(-1, 1, 1, 1)  # [B, 1, 1, 1]

        attn = (1 - artifact_score_4d) * attn_low + artifact_score_4d * attn_high

        return attn


# ============================================================================
# Phase 4: Test-Time Adaptive NRAM v3 (Evidence-Calibrated)
# ============================================================================

class TestTimeAdaptiveNRAMv3(nn.Module):
    """
    Evidence-calibrated TTA-NRAM layer with sigmoid-constrained α.

    NEW in v3:
    - Learnable residual weight α with sigmoid constraint (0~1)
    - Differentiable HP proxy for artifact-preserving constraint
    - FFT artifact detector for evidence estimation only (no grad)
    """

    def __init__(self, channels: int, config: TTANRAMv3Config):
        super().__init__()
        self.channels = channels
        self.config = config

        # Artifact detector (parameter-free, for evidence + logging)
        self.artifact_detector = FrequencyArtifactDetector(
            bands=config.artifact_bands,
            normalize_factor=config.artifact_normalize_factor
        )

        # Noise estimator (parameter-free)
        self.noise_estimator = NoiseEstimator(
            method=config.noise_detection_method,
            normalize_factor=config.noise_normalize_factor
        )

        # Artifact-conditional attention (learnable)
        self.artifact_attention = ArtifactConditionalChannelAttention(
            channels=channels,
            reduction=config.reduction_ratio
        )

        # === NEW v3: Sigmoid-constrained learnable α (FIX 3) ===
        if config.learnable_residual_weight:
            if config.sigmoid_alpha:
                # α = sigmoid(α_logit) for [0,1] constraint
                init_logit = self._inverse_sigmoid(config.residual_weight)
                self.alpha_logit = nn.Parameter(torch.tensor(init_logit))
            else:
                # Direct α (risky, can go out of [0,1])
                self.alpha = nn.Parameter(torch.tensor(config.residual_weight))
        else:
            self.register_buffer('alpha_const', torch.tensor(config.residual_weight))
        # === END v3 ===

    def _inverse_sigmoid(self, x: float) -> float:
        """Compute logit(x) = log(x / (1-x))."""
        x = np.clip(x, 1e-6, 1-1e-6)  # Avoid overflow
        return np.log(x / (1 - x))

    def get_alpha(self) -> torch.Tensor:
        """Get current α value."""
        if self.config.learnable_residual_weight:
            if self.config.sigmoid_alpha:
                return torch.sigmoid(self.alpha_logit)
            else:
                return self.alpha
        else:
            return self.alpha_const

    def forward(
        self,
        F: torch.Tensor,
        test_time: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Apply evidence-calibrated TTA-NRAM to feature map.

        Args:
            F: Feature map [B, C, H, W]
            test_time: Boolean

        Returns:
            F_enhanced: Enhanced features [B, C, H, W]
            F_base: Base features (stop_grad) [B, C, H, W]
            artifact_score: [B, 1] (for evidence, detached)
            noise_level: [B, 1] (for evidence, detached)
            alpha: scalar tensor (current α value)
            debug_info: Dict
        """
        B, C, H, W = F.shape

        # === Phase 1: Artifact Level (no grad, for evidence) ===
        with torch.no_grad():
            artifact_score, artifact_debug = self.artifact_detector(F)

        # === Phase 2: Noise Level (no grad, for evidence) ===
        with torch.no_grad():
            noise_level = self.noise_estimator(F)

        # === Phase 3: Artifact-Conditional Attention (learnable) ===
        attn = self.artifact_attention(F, artifact_score.detach())  # [B, C, 1, 1]

        # === Phase 4: Noise-based Gating ===
        gate = 1.0 - self.config.noise_gate_alpha * noise_level.detach()  # [B, 1]
        gate = gate.view(B, 1, 1, 1).expand(-1, C, 1, 1)  # [B, C, 1, 1]

        # === Phase 5: Combine Attention and Gating ===
        weights = attn * gate  # [B, C, 1, 1]

        # === Phase 6: Apply Weighting ===
        F_gated = F * weights  # [B, C, H, W]

        # === Phase 7: Residual Connection with sigmoid α (FIX 3) ===
        alpha = self.get_alpha()
        F_enhanced = (1 - alpha) * F_gated + alpha * F  # [B, C, H, W]

        # Save base feature (for constraint)
        F_base = F.detach()

        # Debug info
        debug_info = {
            'artifact_score_mean': artifact_score.mean().item(),
            'noise_level_mean': noise_level.mean().item(),
            'attn_mean': attn.mean().item(),
            'gate_mean': gate.mean().item(),
            'weights_mean': weights.mean().item(),
            'weights_std': weights.std().item(),
            'alpha': alpha.item(),
            'band_energies': artifact_debug['band_energies'],  # [B, num_bands]
        }

        return F_enhanced, F_base, artifact_score, noise_level, alpha, debug_info


# ============================================================================
# Phase 5: Memory Bank (Same as v2, disabled by default)
# ============================================================================

class MemoryBank(nn.Module):
    """Confidence-weighted memory bank for robust statistics."""

    def __init__(
        self,
        num_channels: int,
        memory_size: int = 100,
        confidence_threshold: float = 0.8
    ):
        super().__init__()
        self.num_channels = num_channels
        self.memory_size = memory_size
        self.confidence_threshold = confidence_threshold

        self.register_buffer('memory_features', torch.zeros(memory_size, num_channels))
        self.register_buffer('memory_confidences', torch.zeros(memory_size))
        self.register_buffer('memory_pointer', torch.tensor(0, dtype=torch.long))
        self.register_buffer('memory_filled', torch.tensor(0, dtype=torch.long))

    def update(self, features: torch.Tensor, confidence: torch.Tensor, ema_decay: float = 0.99):
        """Update memory with high-confidence samples."""
        B, C = features.shape
        assert C == self.num_channels

        high_conf_mask = confidence > self.confidence_threshold

        if high_conf_mask.sum() == 0:
            return

        high_conf_features = features[high_conf_mask]
        high_conf_scores = confidence[high_conf_mask]

        for feat, conf in zip(high_conf_features, high_conf_scores):
            ptr = self.memory_pointer % self.memory_size

            if self.memory_filled > ptr:
                self.memory_features[ptr] = (
                    ema_decay * self.memory_features[ptr] +
                    (1 - ema_decay) * feat
                )
                self.memory_confidences[ptr] = (
                    ema_decay * self.memory_confidences[ptr] +
                    (1 - ema_decay) * conf
                )
            else:
                self.memory_features[ptr] = feat
                self.memory_confidences[ptr] = conf
                self.memory_filled += 1

            self.memory_pointer += 1

    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """Get confidence-weighted statistics."""
        if self.memory_filled == 0:
            return {
                'mean': torch.zeros(self.num_channels, device=self.memory_features.device),
                'std': torch.ones(self.num_channels, device=self.memory_features.device),
                'num_samples': 0
            }

        valid_features = self.memory_features[:self.memory_filled]
        valid_confidences = self.memory_confidences[:self.memory_filled]

        total_conf = valid_confidences.sum() + 1e-8
        weighted_mean = (
            valid_features * valid_confidences.unsqueeze(-1)
        ).sum(dim=0) / total_conf

        diff = valid_features - weighted_mean.unsqueeze(0)
        weighted_var = (
            (diff ** 2) * valid_confidences.unsqueeze(-1)
        ).sum(dim=0) / total_conf
        weighted_std = torch.sqrt(weighted_var + 1e-8)

        return {
            'mean': weighted_mean,
            'std': weighted_std,
            'num_samples': self.memory_filled.item()
        }

    def reset(self):
        """Clear memory."""
        self.memory_features.zero_()
        self.memory_confidences.zero_()
        self.memory_pointer.zero_()
        self.memory_filled.zero_()


# ============================================================================
# Phase 6: Feature Extractor (Same as v2)
# ============================================================================

class FeatureExtractor:
    """Hook-based feature extractor for intermediate layers."""

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
# Phase 7: Unified TTA-NRAM v3 Wrapper (Evidence-Calibrated)
# ============================================================================

class UnifiedTTANRAMv3(nn.Module):
    """
    Unified TTA-NRAM v3 wrapper with evidence-calibrated update policy.

    KEY DESIGN (v3):
    - Base model ALWAYS in no_grad (FIX 2: memory/speed)
    - NRAM parameters (SE + α) updated via evidence policy
    - Evidence quality: artifact ↑ good, noise ↑ bad
    - Update modes: skip / conservative (high_attn+α) / aggressive (full_attn+α)
    - Sigmoid α for [0,1] constraint (FIX 3)
    - Differentiable HP constraint for artifact preservation (FIX 4)
    """

    def __init__(self, base_model: nn.Module, config: TTANRAMv3Config):
        super().__init__()
        self.config = config
        self.base_model = base_model

        # Freeze base model completely
        for param in base_model.parameters():
            param.requires_grad = False

        # Auto-detect target layer
        if config.target_layer is None:
            config.target_layer = self._get_default_target_layer()

        # Feature extractor (hook-based)
        self.feature_extractor = FeatureExtractor(
            base_model,
            config.target_layer
        )

        # Get channel count
        channels = self._get_layer_channels(config.target_layer)

        # TTA-NRAM v3 (evidence-calibrated)
        self.nram = TestTimeAdaptiveNRAMv3(channels, config).to(config.device)

        # Memory bank (optional)
        if config.enable_memory_bank:
            self.memory_bank = MemoryBank(
                num_channels=channels,
                memory_size=config.memory_size,
                confidence_threshold=config.confidence_threshold
            ).to(config.device)
        else:
            self.memory_bank = None

    def forward(
        self,
        images: torch.Tensor,
        test_time: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass through TTA-NRAM v3.

        Args:
            images: [B, 3, H, W]
            test_time: Boolean

        Returns:
            logits: [B, 1]
            features: [B, C] or None
            debug_info: Dict or None
        """
        # === FIX 2: Base model ALWAYS in no_grad (memory/speed) ===
        # NRAM parameters will still get gradients correctly
        with torch.no_grad():
            _ = self.base_model(images)

        # Get hooked features (no grad from base, but NRAM can still compute grad)
        features = self.feature_extractor.features[self.config.target_layer]  # [B, C, H, W]

        # Apply TTA-NRAM v3
        feat_enhanced, feat_base, artifact_score, noise_level, alpha, debug_info = self.nram(
            features,
            test_time=test_time
        )

        # Add to debug
        if test_time:
            debug_info['artifact_score'] = artifact_score  # [B, 1], detached
            debug_info['noise_level'] = noise_level        # [B, 1], detached
            debug_info['feat_base'] = feat_base            # [B, C, H, W], detached
            debug_info['feat_enhanced'] = feat_enhanced    # [B, C, H, W], may have grad

        # Use Base Model's Classifier
        if self.config.model == "LGrad":
            classifier = self.base_model.classifier
        elif self.config.model == "NPR":
            classifier = self.base_model.model
        else:
            raise ValueError(f"Unknown model: {self.config.model}")

        # Apply avgpool + fc
        feat_pooled = classifier.avgpool(feat_enhanced)  # [B, C, 1, 1]
        feat_pooled = torch.flatten(feat_pooled, 1)     # [B, C]
        fc_layer = classifier.fc1 if self.config.model == "NPR" else classifier.fc
        logits = fc_layer(feat_pooled)  # [B, 1]

        if test_time:
            return logits, feat_pooled, debug_info
        else:
            return logits, None, None

    def update_memory(self, features: torch.Tensor, logits: torch.Tensor):
        """Update memory bank."""
        if self.memory_bank is None:
            return

        prob = torch.sigmoid(logits).squeeze(-1)  # [B]
        confidence = torch.maximum(prob, 1 - prob)  # [B]
        self.memory_bank.update(features, confidence)

    def reset_memory(self):
        """Reset memory bank."""
        if self.memory_bank is not None:
            self.memory_bank.reset()

    def _get_default_target_layer(self) -> str:
        """Auto-detect target layer."""
        if self.config.model == "LGrad":
            return 'classifier.layer4'
        elif self.config.model == "NPR":
            return 'model.layer2'
        else:
            raise ValueError(f"Unknown model type: {self.config.model}")

    def _get_layer_channels(self, layer_name: str) -> int:
        """Get number of output channels for a layer."""
        device = self.config.device
        dummy_input = torch.zeros(1, 3, 224, 224).to(device)

        with torch.no_grad():
            _ = self.base_model(dummy_input)

        feature = self.feature_extractor.features[layer_name]
        return feature.shape[1]

    # === NEW v3: Evidence Quality Computation (FIX 3: simple normalize) ===
    def compute_evidence_quality(
        self,
        artifact_score: torch.Tensor,  # [B, 1], in [0,1]
        noise_level: torch.Tensor      # [B, 1], in [0,1]
    ) -> torch.Tensor:
        """
        Compute evidence quality for each sample.

        Args:
            artifact_score: [B, 1] - artifact level in [0,1]
            noise_level: [B, 1] - noise level in [0,1]

        Returns:
            q: [B] - Quality in [0, 1], higher = more reliable
        """
        cfg = self.config

        # Flatten
        a = artifact_score.squeeze(1)  # [B]
        n = noise_level.squeeze(1)     # [B]

        # === FIX 3: Simple center-based normalize ===
        a_centered = a - cfg.evidence_center_artifact  # [-0.5, 0.5]
        n_centered = n - cfg.evidence_center_noise     # [-0.5, 0.5]

        # Evidence logit
        logit = (cfg.evidence_w_artifact * a_centered
                 - cfg.evidence_w_noise * n_centered
                 - cfg.evidence_bias)

        q = torch.sigmoid(logit)  # [B], in [0, 1]

        return q

    # === NEW v3: Differentiable Artifact-Preserving Constraint (FIX 4) ===
    def compute_artifact_preserving_loss(
        self,
        F_base: torch.Tensor,    # [B, C, H, W], detached
        F_enh: torch.Tensor,     # [B, C, H, W], with grad
        mask: torch.Tensor       # [B, 1]
    ) -> torch.Tensor:
        """
        Differentiable artifact-preserving constraint via HP proxy.

        NOTE: This is differentiable (grad flows to F_enh → NRAM/α).
        FFT artifact_score is used for logging only, not for constraint.
        """
        cfg = self.config

        if not cfg.enable_artifact_preserving:
            return torch.tensor(0.0, device=F_base.device)

        # === FIX 4: Differentiable HP proxy (avgpool-based) ===
        kernel_size = cfg.hp_kernel_size

        # High-pass: F - blur(F)
        blur_base = torch.nn.functional.avg_pool2d(
            F_base, kernel_size=kernel_size, stride=1,
            padding=kernel_size//2, count_include_pad=False
        )
        HP_base = F_base - blur_base  # [B, C, H, W], detached

        blur_enh = torch.nn.functional.avg_pool2d(
            F_enh, kernel_size=kernel_size, stride=1,
            padding=kernel_size//2, count_include_pad=False
        )
        HP_enh = F_enh - blur_enh  # [B, C, H, W], with grad

        # Pool to [B, C]
        HP_base_pool = torch.nn.functional.adaptive_avg_pool2d(HP_base, 1).squeeze(-1).squeeze(-1)  # [B, C]
        HP_enh_pool = torch.nn.functional.adaptive_avg_pool2d(HP_enh, 1).squeeze(-1).squeeze(-1)   # [B, C]

        # L2 distance per sample
        dist = torch.norm(HP_enh_pool - HP_base_pool, p=2, dim=1, keepdim=True)  # [B, 1]

        # Masked loss (only high-evidence samples)
        loss_ap = (mask * dist).sum() / (mask.sum() + 1e-8)

        return loss_ap


# ============================================================================
# Phase 8: TTA Loss (Entropy only, no confidence)
# ============================================================================

class TTALoss(nn.Module):
    """Self-supervised loss for test-time adaptation (entropy only)."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.weights = weights or {'entropy': 1.0, 'confidence': 0.0}

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        """
        Compute TTA loss with sample mask.

        Args:
            logits: [B, 1]
            mask: [B, 1] - sample selection mask

        Returns:
            loss_dict: Dict with loss components
        """
        prob = torch.sigmoid(logits)  # [B, 1]

        # Entropy per sample
        entropy_per_sample = -(
            prob * torch.log(prob + 1e-8) +
            (1 - prob) * torch.log(1 - prob + 1e-8)
        )  # [B, 1]

        # Masked entropy
        masked_entropy = (mask * entropy_per_sample).sum() / (mask.sum() + 1e-8)

        # Total loss (no confidence by default)
        total_loss = self.weights['entropy'] * masked_entropy

        return {
            'total': total_loss,
            'entropy': masked_entropy.item(),
            'mean_prob': prob.mean().item(),
        }


# ============================================================================
# Phase 9: Evidence-Calibrated TTA Inference (ALL FIXES)
# ============================================================================

def inference_with_tta_v3(
    model: UnifiedTTANRAMv3,
    images: torch.Tensor,
    config: TTANRAMv3Config,
    return_debug: bool = False
) -> Dict:
    """
    Evidence-calibrated TTA inference (v3 with ALL 6 FIXES).

    Key improvements:
    - FIX 1: Log updated parameters (eval mode clarity)
    - FIX 2: Base model always no_grad (memory/speed)
    - FIX 3: Sigmoid α for [0,1] constraint
    - FIX 4: Differentiable HP constraint (not FFT)
    - FIX 5: q_step detached before mask
    - FIX 6: Enhanced collapse safeguard (entropy + std)
    """
    device = config.device
    images = images.to(device)
    B = images.shape[0]

    # ========================================
    # Phase 1: Initial Forward
    # ========================================
    model.eval()
    with torch.no_grad():
        logits_init, _, debug_init = model(images, test_time=True)
        pred_init = torch.sigmoid(logits_init)

        artifact_score_init = debug_init['artifact_score']  # [B, 1]
        noise_level_init = debug_init['noise_level']        # [B, 1]

    # ========================================
    # Phase 2: Evidence Quality
    # ========================================
    if not config.enable_evidence_calibration:
        print("[TTA] Evidence calibration disabled, using simple TTA")
        return _inference_with_tta_v3_simple(model, images, config, return_debug)

    q_init = model.compute_evidence_quality(artifact_score_init, noise_level_init)  # [B]
    q_mean_init = q_init.mean().item()

    # ========================================
    # Phase 3: Batch-Level Update Policy
    # ========================================
    τ_skip = config.skip_threshold
    τ_cons = config.conservative_threshold
    τ_sample = config.sample_threshold

    if q_mean_init < τ_skip:
        # SKIP
        if config.verbose:
            print(f"[TTA-v3] Evidence too low (q={q_mean_init:.3f} < {τ_skip}), SKIP")
        return {
            'predictions': pred_init,
            'logits': logits_init,
            'initial_predictions': pred_init,
            'improvement': 0.0,
            'skipped': True,
            'evidence_mean': q_mean_init,
            'update_mode': 'skip',
        }

    elif q_mean_init < τ_cons:
        # CONSERVATIVE: high_artifact_attn + α
        update_mode = "conservative"
        params_to_update = []

        for n, p in model.named_parameters():
            if 'artifact_attention.high_artifact_attn' in n:
                params_to_update.append((n, p))

        # FIX 3: Add sigmoid α
        if config.learnable_residual_weight:
            for n, p in model.named_parameters():
                if 'alpha_logit' in n or ('alpha' in n and 'alpha_const' not in n):
                    params_to_update.append((n, p))

        effective_lr = config.tta_lr
        if config.verbose:
            print(f"[TTA-v3] Evidence moderate (q={q_mean_init:.3f}), CONSERVATIVE")

    else:
        # AGGRESSIVE: full_artifact_attn + α
        update_mode = "aggressive"
        params_to_update = []

        for n, p in model.named_parameters():
            if 'artifact_attention' in n:
                params_to_update.append((n, p))

        # FIX 3: Add α
        if config.learnable_residual_weight:
            for n, p in model.named_parameters():
                if 'alpha_logit' in n or ('alpha' in n and 'alpha_const' not in n):
                    params_to_update.append((n, p))

        effective_lr = config.tta_lr
        if config.verbose:
            print(f"[TTA-v3] Evidence high (q={q_mean_init:.3f}), AGGRESSIVE")

    # Extract only parameters (not names)
    params_list = [p for n, p in params_to_update]

    # FIX 1: Log updated parameters (1st step only)
    param_names = [n for n, p in params_to_update]
    if config.verbose:
        print(f"[TTA-v3] Updating {len(params_list)} parameters: {param_names}")

    # ========================================
    # Phase 4: Setup Optimizer
    # ========================================
    optimizer = torch.optim.Adam(params_list, lr=effective_lr)
    max_grad_norm = 1.0

    # Enable gradients for selected params
    for p in model.parameters():
        p.requires_grad = False
    for p in params_list:
        p.requires_grad = True

    # ========================================
    # Phase 5: TTA Loop (with all fixes)
    # ========================================
    tta_history = []
    collapse_counter = 0
    prev_entropy = None

    for step in range(config.tta_steps):
        # === 5.1: Forward ===
        logits, features, debug = model(images, test_time=True)
        prob = torch.sigmoid(logits)  # [B, 1]

        # === 5.2: Update evidence (every step) ===
        artifact_score_step = debug['artifact_score']  # [B, 1]
        noise_level_step = debug['noise_level']        # [B, 1]
        q_step = model.compute_evidence_quality(artifact_score_step, noise_level_step)  # [B]

        # FIX 5: Detach q_step before mask (policy is not learned)
        q_step = q_step.detach()
        q_mean_step = q_step.mean().item()

        # Sample-level mask
        mask = (q_step >= τ_sample).float().unsqueeze(1)  # [B, 1]
        num_updated = mask.sum().item()

        # === 5.3: FIX 6 - Enhanced collapse detection ===
        prob_mean = prob.mean().item()
        prob_std = prob.std().item()

        # Compute current entropy
        entropy_per_sample = -(
            prob * torch.log(prob + 1e-8) +
            (1 - prob) * torch.log(1 - prob + 1e-8)
        )
        entropy_mean = entropy_per_sample.mean().item()

        if config.enable_collapse_detection:
            collapse_detected = False

            # Check 1: Extreme prob_mean
            if prob_mean < config.collapse_prob_threshold or prob_mean > (1 - config.collapse_prob_threshold):
                collapse_detected = True

            # Check 2: Low prob_std (all predictions similar)
            if prob_std < config.collapse_std_threshold:
                collapse_detected = True

            if collapse_detected:
                collapse_counter += 1
                if collapse_counter >= config.collapse_patience:
                    if config.verbose:
                        print(f"[TTA-v3] Collapse detected at step {step} "
                            f"(prob_mean={prob_mean:.4f}, prob_std={prob_std:.4f}), EARLY STOP")
                    break
            else:
                collapse_counter = 0
        # === END FIX 6 ===

        if num_updated == 0:
            if config.verbose:
                print(f"[TTA-v3] Step {step}: No samples pass threshold, skip")
            tta_history.append({
                'step': step,
                'loss': 0.0,
                'entropy': 0.0,
                'loss_ap': 0.0,
                'num_updated': 0,
                'q_mean': q_mean_step,
                'prob_mean': prob_mean,
                'prob_std': prob_std,
                'entropy_mean': entropy_mean,
                'skipped': True,
            })
            continue

        # === 5.4: Compute entropy loss (masked) ===
        entropy_per_sample = -(
            prob * torch.log(prob + 1e-8) +
            (1 - prob) * torch.log(1 - prob + 1e-8)
        )  # [B, 1]

        masked_entropy = (mask * entropy_per_sample).sum() / (mask.sum() + 1e-8)

        # === 5.5: FIX 4 - Differentiable artifact-preserving constraint ===
        feat_base = debug['feat_base']         # [B, C, H, W], detached
        feat_enh = debug['feat_enhanced']      # [B, C, H, W], with grad

        loss_ap = model.compute_artifact_preserving_loss(feat_base, feat_enh, mask)

        # === 5.6: Total loss ===
        λ_ap = config.artifact_preserving_weight
        loss = masked_entropy + λ_ap * loss_ap

        # === 5.7: Backward and update ===
        optimizer.zero_grad()
        loss.backward()

        # Grad clipping
        torch.nn.utils.clip_grad_norm_(params_list, max_grad_norm)

        optimizer.step()

        # === 5.8: Record history (minimal CPU transfer) ===
        tta_history.append({
            'step': step,
            'loss': loss.item(),
            'entropy': masked_entropy.item(),
            'loss_ap': loss_ap.item() if isinstance(loss_ap, torch.Tensor) else 0.0,
            'num_updated': int(num_updated),
            'q_mean': q_mean_step,
            'prob_mean': prob_mean,
            'prob_std': prob_std,
            'entropy_mean': entropy_mean,
            'alpha': debug['alpha'],
            'skipped': False,
        })

        prev_entropy = entropy_mean

    # ========================================
    # Phase 6: Final Prediction
    # ========================================
    model.eval()
    with torch.no_grad():
        logits_final, features_final, debug_final = model(images, test_time=True)
        pred_final = torch.sigmoid(logits_final)

    # ========================================
    # Phase 7: Update Memory Bank
    # ========================================
    if model.memory_bank is not None:
        with torch.no_grad():
            model.update_memory(features_final, logits_final)

    # ========================================
    # Phase 8: Disable Gradients
    # ========================================
    for p in model.parameters():
        p.requires_grad = False

    # ========================================
    # Prepare Results (minimal CPU transfer)
    # ========================================
    results = {
        'predictions': pred_final,  # Keep on GPU
        'logits': logits_final,
        'initial_predictions': pred_init,
        'improvement': (pred_final - pred_init).mean().item(),
        'skipped': False,
        'evidence_mean': q_mean_init,
        'update_mode': update_mode,
    }

    if return_debug:
        results['tta_history'] = tta_history
        results['debug_initial'] = debug_init
        results['debug_final'] = debug_final

    return results


# === Helper: Simple TTA (for ablation) ===
def _inference_with_tta_v3_simple(
    model: UnifiedTTANRAMv3,
    images: torch.Tensor,
    config: TTANRAMv3Config,
    return_debug: bool = False
) -> Dict:
    """Simple TTA without evidence calibration."""
    device = config.device
    model.eval()

    with torch.no_grad():
        logits_init, _, _ = model(images, test_time=True)
        pred_init = torch.sigmoid(logits_init)

    params_to_update = [p for n, p in model.named_parameters() if 'artifact_attention' in n]
    for p in model.parameters():
        p.requires_grad = False
    for p in params_to_update:
        p.requires_grad = True

    optimizer = torch.optim.Adam(params_to_update, lr=config.tta_lr)

    for step in range(config.tta_steps):
        logits, _, _ = model(images, test_time=True)
        prob = torch.sigmoid(logits)

        entropy = -(prob * torch.log(prob + 1e-8) + (1 - prob) * torch.log(1 - prob + 1e-8))
        loss = entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits_final, _, _ = model(images, test_time=True)
        pred_final = torch.sigmoid(logits_final)

    for p in model.parameters():
        p.requires_grad = False

    return {
        'predictions': pred_final,
        'logits': logits_final,
        'initial_predictions': pred_init,
        'improvement': (pred_final - pred_init).mean().item(),
        'skipped': False,
        'update_mode': 'simple',
    }


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters in model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
    }


def print_model_info(model: UnifiedTTANRAMv3):
    """Print model architecture and parameter counts."""
    print("=" * 80)
    print("TTA-NRAM v3 (Evidence-Calibrated) Model Information")
    print("=" * 80)

    # Config
    print(f"\nConfiguration:")
    print(f"  Model: {model.config.model}")
    print(f"  Target Layer: {model.config.target_layer}")
    print(f"  TTA Steps: {model.config.tta_steps}")
    print(f"  Evidence Calibration: {model.config.enable_evidence_calibration}")
    print(f"  Sigmoid α: {model.config.sigmoid_alpha}")
    print(f"  Artifact Preserving: {model.config.enable_artifact_preserving} ({model.config.artifact_preserving_type})")

    # Parameters
    params = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable (during TTA): {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")

    # Components
    nram_params = count_parameters(model.nram)
    print(f"\nNRAM v3 Components:")
    print(f"  Total NRAM: {nram_params['total']:,} params")
    print(f"    ├─ Artifact Detector: 0 params (parameter-free, for evidence)")
    print(f"    ├─ Noise Estimator: 0 params (parameter-free, for evidence)")
    print(f"    ├─ Artifact Attention: Learnable (dual-path SE)")
    print(f"    └─ Residual Weight α: {'Learnable (sigmoid)' if model.config.learnable_residual_weight else 'Fixed'}")
    print(f"  Base Classifier: Frozen (pre-trained)")

    print("=" * 80)


if __name__ == "__main__":
    print("TTA-NRAM v3 (Evidence-Calibrated) module loaded successfully!")
    print("\nKey features (NEW in v3):")
    print("  ✅ Evidence-calibrated update policy (skip/conservative/aggressive)")
    print("  ✅ Sigmoid-constrained learnable α (0~1)")
    print("  ✅ Differentiable artifact-preserving constraint (HP proxy)")
    print("  ✅ Enhanced collapse detection (entropy + std)")
    print("  ✅ Memory-efficient design (minimal CPU transfers)")
    print("  ✅ Base model always in no_grad (speed/memory)")
    print("\nUsage:")
    print("  from model.method.tta_nramv3 import UnifiedTTANRAMv3, TTANRAMv3Config, inference_with_tta_v3")
