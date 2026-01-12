"""
Test-Time Adaptive NRAM v2 (Artifact-Aware) for Deepfake Detection

This module implements an artifact-aware test-time adaptation approach that:
1. Detects GAN artifacts in frequency domain (even under corruption)
2. Applies artifact-conditional channel attention
3. Preserves deepfake-specific signals while handling noise

Key Improvements over v1:
- Frequency-domain artifact detection (parameter-free)
- Dual-path channel attention (low-artifact vs high-artifact)
- Artifact-aware feature enhancement
- Better robustness under Gaussian/JPEG corruptions

Architecture:
    Base Model (frozen) → layer4 features
        ↓
    Artifact Detector (frequency-based)
        ↓
    Artifact-Conditional Attention (dual-path)
        ↓
    Noise-based Gating
        ↓
    Enhanced features → Base Classifier
        ↓
    Final prediction

Author: [Your name]
Date: 2026-01-11
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
class TTANRAMv2Config:
    """Configuration for TTA-NRAM v2 (Artifact-Aware)"""

    # Model selection
    model: str = "LGrad"  # "LGrad" or "NPR"

    # Target layer
    target_layer: str = None  # e.g., 'classifier.layer4' (auto-detected if None)

    # Channel attention
    reduction_ratio: int = 16  # SE-Net style reduction

    # Artifact detection (NEW in v2)
    artifact_bands: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.2, 0.4), (0.4, 0.6)]  # Frequency bands for artifact
    )
    artifact_normalize_factor: float = 1.0  # Scale artifact score

    # Noise estimation
    noise_detection_method: str = "laplacian"  # "laplacian" or "variance"
    noise_normalize_factor: float = 100.0  # Scale variance to [0,1]
    noise_gate_alpha: float = 1.0  # gate = 1 - alpha * noise_level

    # Memory bank
    enable_memory_bank: bool = True
    memory_size: int = 100
    confidence_threshold: float = 0.8  # Only store high-confidence samples

    # TTA settings
    tta_steps: int = 5
    tta_lr: float = 1e-4  # Learning rate for TTA updates
    tta_loss_weights: Dict[str, float] = field(
        default_factory=lambda: {"entropy": 1.0, "confidence": 0.1}
    )

    # Gating
    residual_weight: float = 0.1  # F_out = (1-α)*F_gated + α*F_in

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Phase 1: Frequency-based Artifact Detector (NEW in v2)
# ============================================================================

class FrequencyArtifactDetector(nn.Module):
    """
    Parameter-free artifact detection using frequency-domain analysis.

    GAN artifacts (e.g., upsampling artifacts) leave characteristic signatures
    in the frequency domain that persist even under Gaussian/JPEG corruption.

    Process:
    1. Convert feature map to frequency domain (2D FFT)
    2. Compute magnitude spectrum
    3. Extract energy in artifact-sensitive bands
    4. Compare with baseline to get artifact score

    Args:
        bands: List of (low, high) frequency band ratios in [0,1]
               e.g., [(0.2, 0.4), (0.4, 0.6)] for mid-high frequencies
        normalize_factor: Scale for final score

    Returns:
        artifact_score: [B, 1] tensor in range [0, 1]
            - 0: Low artifact (real-like)
            - 1: High artifact (fake-like)
    """

    def __init__(
        self,
        bands: List[Tuple[float, float]] = None,
        normalize_factor: float = 1.0
    ):
        super().__init__()
        self.bands = bands or [(0.2, 0.4), (0.4, 0.6)]
        self.normalize_factor = normalize_factor

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Detect artifact level from feature map.

        Args:
            feature_map: [B, C, H, W]

        Returns:
            artifact_score: [B, 1]
        """
        B, C, H, W = feature_map.shape
        device = feature_map.device

        # ========================================
        # Step 1: Channel-wise average (reduce to [B, 1, H, W])
        # ========================================
        F_mean = feature_map.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # ========================================
        # Step 2: 2D FFT → Magnitude Spectrum
        # ========================================
        # Apply FFT to spatial dimensions
        F_freq = torch.fft.fft2(F_mean.squeeze(1))  # [B, H, W] complex
        F_mag = torch.abs(F_freq)  # [B, H, W]

        # Shift zero frequency to center
        F_mag = torch.fft.fftshift(F_mag, dim=(-2, -1))  # [B, H, W]

        # ========================================
        # Step 3: Create radial frequency mask
        # ========================================
        # Coordinate grid centered at (H/2, W/2)
        y, x = torch.meshgrid(
            torch.arange(H, device=device) - H // 2,
            torch.arange(W, device=device) - W // 2,
            indexing='ij'
        )

        # Radial distance (normalized to [0, 1])
        max_radius = np.sqrt((H // 2) ** 2 + (W // 2) ** 2)
        radius = torch.sqrt(x.float() ** 2 + y.float() ** 2) / max_radius  # [H, W]

        # ========================================
        # Step 4: Extract band energies
        # ========================================
        band_scores = []

        for low, high in self.bands:
            # Create mask for this band
            mask = ((radius >= low) & (radius < high)).float()  # [H, W]

            # Average magnitude in band
            band_mag = (F_mag * mask.unsqueeze(0)).sum(dim=(-2, -1))  # [B]
            band_count = mask.sum() + 1e-8
            band_mag = band_mag / band_count  # Normalize by number of pixels

            band_scores.append(band_mag)

        # Average across bands
        artifact_raw = torch.stack(band_scores, dim=1).mean(dim=1)  # [B]

        # ========================================
        # Step 5: Compute baseline (low-frequency energy)
        # ========================================
        baseline_mask = (radius < 0.2).float()  # Low frequencies
        baseline_mag = (F_mag * baseline_mask.unsqueeze(0)).sum(dim=(-2, -1))  # [B]
        baseline_count = baseline_mask.sum() + 1e-8
        baseline_mag = baseline_mag / baseline_count + 1e-8

        # ========================================
        # Step 6: Relative score
        # ========================================
        # artifact_score = (band_mag - baseline) / baseline
        relative_score = (artifact_raw - baseline_mag) / baseline_mag  # [B]

        # Normalize and clamp
        artifact_score = torch.sigmoid(relative_score * self.normalize_factor)  # [B]
        artifact_score = artifact_score.unsqueeze(1)  # [B, 1]

        return artifact_score


# ============================================================================
# Phase 2: Noise Estimator (Same as v1)
# ============================================================================

class NoiseEstimator(nn.Module):
    """
    Parameter-free noise level estimation based on high-frequency analysis.
    (Same as v1, copied for completeness)
    """

    def __init__(self, method: str = "laplacian", normalize_factor: float = 100.0):
        super().__init__()
        self.method = method
        self.normalize_factor = normalize_factor

        if method == "laplacian":
            # Laplacian kernel
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

        # Reshape and apply convolution
        F_reshaped = F.view(B * C, 1, H, W)
        F_filtered = torch.nn.functional.conv2d(F_reshaped, kernel, padding=1)  # [B*C, 1, H, W]
        F_filtered = F_filtered.view(B, C, H, W)

        # Compute variance
        spatial_var = F_filtered.var(dim=[2, 3])  # [B, C]
        noise_var = spatial_var.mean(dim=1, keepdim=True)  # [B, 1]

        # Normalize to [0, 1]
        noise_level = torch.clamp(noise_var / self.normalize_factor, 0.0, 1.0)

        return noise_level

    def _estimate_variance(self, F: torch.Tensor) -> torch.Tensor:
        """Simple spatial variance-based noise estimation."""
        B, C, H, W = F.shape

        # Spatial variance per channel
        spatial_var = F.var(dim=[2, 3])  # [B, C]
        noise_var = spatial_var.mean(dim=1, keepdim=True)  # [B, 1]

        # Normalize to [0, 1]
        noise_level = torch.clamp(noise_var / self.normalize_factor, 0.0, 1.0)

        return noise_level


# ============================================================================
# Phase 3: Artifact-Conditional Channel Attention (NEW in v2)
# ============================================================================

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) style channel attention.
    (Same as v1, used as building block)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, H, W] → [B, C, 1, 1]
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

    Motivation:
    - Low-artifact samples (real-like): Emphasize channels robust to noise
    - High-artifact samples (fake-like): Emphasize channels sensitive to artifacts

    Architecture:
        Two SE-Net branches:
        - low_artifact_attn: For real-like samples
        - high_artifact_attn: For fake-like samples

        Interpolation based on artifact_score:
        attn = (1 - artifact_score) * attn_low + artifact_score * attn_high

    Args:
        channels: Number of input channels (e.g., 2048)
        reduction: Reduction ratio for SE bottleneck

    Returns:
        attn_weights: [B, C, 1, 1] artifact-conditional attention
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels

        # Dual paths
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
        # Compute both paths
        attn_low = self.low_artifact_attn(F)   # [B, C, 1, 1]
        attn_high = self.high_artifact_attn(F)  # [B, C, 1, 1]

        # Interpolate based on artifact score
        artifact_score_4d = artifact_score.view(-1, 1, 1, 1)  # [B, 1, 1, 1]

        attn = (1 - artifact_score_4d) * attn_low + artifact_score_4d * attn_high

        return attn


# ============================================================================
# Phase 4: Test-Time Adaptive NRAM v2 (Artifact-Aware)
# ============================================================================

class TestTimeAdaptiveNRAMv2(nn.Module):
    """
    Artifact-aware TTA-NRAM layer for adaptive channel gating.

    NEW in v2:
    - Artifact detection in frequency domain
    - Artifact-conditional dual-path attention
    - Separate gating for noise vs artifact

    Process:
    1. Estimate artifact level (frequency-based, parameter-free)
    2. Estimate noise level (Laplacian-based, parameter-free)
    3. Compute artifact-conditional attention (learnable, dual-path)
    4. Apply noise-based gating
    5. Combine: weights = attention × gate
    6. Enhance features with residual connection

    Args:
        channels: Number of input channels (e.g., 2048 for ResNet layer4)
        config: TTANRAMv2Config
    """

    def __init__(self, channels: int, config: TTANRAMv2Config):
        super().__init__()
        self.channels = channels
        self.config = config

        # Artifact detector (parameter-free, NEW in v2)
        self.artifact_detector = FrequencyArtifactDetector(
            bands=config.artifact_bands,
            normalize_factor=config.artifact_normalize_factor
        )

        # Noise estimator (parameter-free, same as v1)
        self.noise_estimator = NoiseEstimator(
            method=config.noise_detection_method,
            normalize_factor=config.noise_normalize_factor
        )

        # Artifact-conditional attention (learnable, NEW in v2)
        self.artifact_attention = ArtifactConditionalChannelAttention(
            channels=channels,
            reduction=config.reduction_ratio
        )

    def forward(
        self,
        F: torch.Tensor,
        test_time: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Apply artifact-aware TTA-NRAM to feature map.

        Args:
            F: Feature map [B, C, H, W]
            test_time: Boolean (True during TTA, False during training)

        Returns:
            F_enhanced: Enhanced features [B, C, H, W]
            weights: Adaptive weights [B, C, 1, 1]
            artifact_score: Estimated artifact level [B, 1]
            noise_level: Estimated noise [B, 1]
            debug_info: Dict with diagnostic information
        """
        B, C, H, W = F.shape

        # ========================================
        # Phase 1: Artifact Level Estimation (NEW)
        # ========================================
        artifact_score = self.artifact_detector(F)  # [B, 1]

        # ========================================
        # Phase 2: Noise Level Estimation
        # ========================================
        noise_level = self.noise_estimator(F)  # [B, 1]

        # ========================================
        # Phase 3: Artifact-Conditional Attention (NEW)
        # ========================================
        attn = self.artifact_attention(F, artifact_score)  # [B, C, 1, 1]

        # ========================================
        # Phase 4: Noise-based Gating
        # ========================================
        # gate = 1 - alpha * noise_level
        # High noise → low gate → suppress
        gate = 1.0 - self.config.noise_gate_alpha * noise_level  # [B, 1]
        gate = gate.view(B, 1, 1, 1).expand(-1, C, 1, 1)  # [B, C, 1, 1]

        # ========================================
        # Phase 5: Combine Attention and Gating
        # ========================================
        # Final weight = attention × gate
        weights = attn * gate  # [B, C, 1, 1]

        # ========================================
        # Phase 6: Apply Weighting
        # ========================================
        F_gated = F * weights  # [B, C, H, W]

        # ========================================
        # Phase 7: Residual Connection (Stability)
        # ========================================
        alpha = self.config.residual_weight
        F_enhanced = (1 - alpha) * F_gated + alpha * F  # [B, C, H, W]

        # ========================================
        # Debug Information
        # ========================================
        debug_info = {
            'artifact_score_mean': artifact_score.mean().item(),
            'noise_level_mean': noise_level.mean().item(),
            'attn_mean': attn.mean().item(),
            'gate_mean': gate.mean().item(),
            'weights_mean': weights.mean().item(),
            'weights_std': weights.std().item(),
        }

        return F_enhanced, weights, artifact_score, noise_level, debug_info


# ============================================================================
# Phase 5: Memory Bank (Same as v1)
# ============================================================================

class MemoryBank(nn.Module):
    """
    Confidence-weighted memory bank for robust statistics.
    (Same as v1, copied for completeness)
    """

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

        # Memory buffers (FIFO queue)
        self.register_buffer(
            'memory_features',
            torch.zeros(memory_size, num_channels)
        )
        self.register_buffer(
            'memory_confidences',
            torch.zeros(memory_size)
        )
        self.register_buffer(
            'memory_pointer',
            torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            'memory_filled',
            torch.tensor(0, dtype=torch.long)
        )

    def update(
        self,
        features: torch.Tensor,
        confidence: torch.Tensor,
        ema_decay: float = 0.99
    ):
        """Update memory with high-confidence samples."""
        B, C = features.shape
        assert C == self.num_channels

        # Filter high-confidence samples
        high_conf_mask = confidence > self.confidence_threshold

        if high_conf_mask.sum() == 0:
            return

        high_conf_features = features[high_conf_mask]
        high_conf_scores = confidence[high_conf_mask]

        # Update memory (FIFO with EMA)
        for feat, conf in zip(high_conf_features, high_conf_scores):
            ptr = self.memory_pointer % self.memory_size

            if self.memory_filled > ptr:
                # EMA blend
                self.memory_features[ptr] = (
                    ema_decay * self.memory_features[ptr] +
                    (1 - ema_decay) * feat
                )
                self.memory_confidences[ptr] = (
                    ema_decay * self.memory_confidences[ptr] +
                    (1 - ema_decay) * conf
                )
            else:
                # First fill
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

        # Confidence-weighted mean
        total_conf = valid_confidences.sum() + 1e-8
        weighted_mean = (
            valid_features * valid_confidences.unsqueeze(-1)
        ).sum(dim=0) / total_conf

        # Confidence-weighted std
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
# Phase 6: Feature Extractor (Same as v1)
# ============================================================================

class FeatureExtractor:
    """
    Hook-based feature extractor for intermediate layers.
    (Same as v1)
    """

    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.features = {}
        self.hooks = []

        # Register hook
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
# Phase 7: Unified TTA-NRAM v2 Wrapper
# ============================================================================

class UnifiedTTANRAMv2(nn.Module):
    """
    Unified TTA-NRAM v2 wrapper for LGrad and NPR models.

    ARCHITECTURE:
        Input Image [B, 3, H, W]
            ↓
        Base Model (frozen) → extract layer4 features
            ↓
        layer4 features [B, 2048, 7, 7]
            ↓
        TestTimeAdaptiveNRAMv2 (artifact-aware gating)
            - Artifact detection (frequency-based)
            - Noise estimation (Laplacian-based)
            - Artifact-conditional attention (dual-path)
            - Adaptive gating
            ↓
        Enhanced features [B, 2048, 7, 7]
            ↓
        Base Classifier (avgpool + fc) - Pre-trained!
            ↓
        Logits [B, 1]

    KEY DESIGN:
    - ✅ Single target layer (layer4 only)
    - ✅ NO adapter training needed - uses pre-trained classifier
    - ✅ Artifact-aware dual-path attention (NEW in v2)
    - ✅ Frequency-domain artifact detection (NEW in v2)
    - ✅ Memory bank for robust statistics

    Args:
        base_model: Pre-trained LGrad or NPR model (will be frozen)
        config: TTANRAMv2Config
    """

    def __init__(self, base_model: nn.Module, config: TTANRAMv2Config):
        super().__init__()
        self.config = config
        self.base_model = base_model

        # Freeze base model completely
        for param in base_model.parameters():
            param.requires_grad = False

        # Auto-detect target layer if not specified
        if config.target_layer is None:
            config.target_layer = self._get_default_target_layer()

        # Feature extractor (hook-based)
        self.feature_extractor = FeatureExtractor(
            base_model,
            config.target_layer
        )

        # Get channel count for target layer
        channels = self._get_layer_channels(config.target_layer)

        # TTA-NRAM v2 module (artifact-aware)
        self.nram = TestTimeAdaptiveNRAMv2(channels, config).to(config.device)

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
        Forward pass through TTA-NRAM v2.

        Args:
            images: [B, 3, H, W]
            test_time: Boolean
                - False: Normal forward (no TTA)
                - True: TTA mode (for test-time adaptation)

        Returns:
            logits: [B, 1] - Binary classification
            features: [B, C] or None - Enhanced features (if test_time=True)
            debug_info: Dict or None - Debug information (if test_time=True)
        """
        # ========================================
        # Step 1: Base Model → Extract Features
        # ========================================
        with torch.no_grad():  # Base model always frozen
            _ = self.base_model(images)

        # Get hooked features
        features = self.feature_extractor.features[self.config.target_layer]  # [B, C, H, W]

        # ========================================
        # Step 2: Apply TTA-NRAM v2 (Artifact-Aware)
        # ========================================
        feat_enhanced, weights, artifact_score, noise_level, debug_info = self.nram(
            features,
            test_time=test_time
        )

        # Add artifact/noise info to debug
        if test_time:
            debug_info['artifact_score'] = artifact_score
            debug_info['noise_level'] = noise_level

        # ========================================
        # Step 3: Use Base Model's Classifier
        # ========================================
        if self.config.model == "LGrad":
            classifier = self.base_model.classifier
        elif self.config.model == "NPR":
            classifier = self.base_model.model
        else:
            raise ValueError(f"Unknown model: {self.config.model}")

        # Apply avgpool
        feat_pooled = classifier.avgpool(feat_enhanced)  # [B, C, 1, 1]
        feat_pooled = torch.flatten(feat_pooled, 1)     # [B, C]

        # Apply fc (final classifier)
        # NPR uses 'fc1' instead of 'fc'
        fc_layer = classifier.fc1 if self.config.model == "NPR" else classifier.fc
        logits = fc_layer(feat_pooled)  # [B, 1]

        # ========================================
        # Return
        # ========================================
        if test_time:
            return logits, feat_pooled, debug_info
        else:
            return logits, None, None

    def update_memory(self, features: torch.Tensor, logits: torch.Tensor):
        """Update memory bank with features and confidence."""
        if self.memory_bank is None:
            warnings.warn("Memory bank is disabled, skipping update")
            return

        # Convert logits to confidence
        prob = torch.sigmoid(logits).squeeze(-1)  # [B]
        confidence = torch.maximum(prob, 1 - prob)  # [B]

        # Update memory
        self.memory_bank.update(features, confidence)

    def reset_memory(self):
        """Reset memory bank."""
        if self.memory_bank is not None:
            self.memory_bank.reset()

    def _get_default_target_layer(self) -> str:
        """Auto-detect target layer based on model type."""
        if self.config.model == "LGrad":
            return 'classifier.layer4'
        elif self.config.model == "NPR":
            return 'model.layer2'  # NPR only uses layer1-2, not layer3-4
        else:
            raise ValueError(f"Unknown model type: {self.config.model}")

    def _get_layer_channels(self, layer_name: str) -> int:
        """Get number of output channels for a layer."""
        device = self.config.device
        dummy_input = torch.zeros(1, 3, 224, 224).to(device)

        with torch.no_grad():
            _ = self.base_model(dummy_input)

        feature = self.feature_extractor.features[layer_name]
        return feature.shape[1]  # C from [B, C, H, W]


# ============================================================================
# Phase 8: TTA Loss (Same as v1)
# ============================================================================

class TTALoss(nn.Module):
    """
    Self-supervised loss for test-time adaptation.
    (Same as v1)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.weights = weights or {'entropy': 1.0, 'confidence': 0.1}

    def forward(self, logits: torch.Tensor) -> Dict[str, float]:
        """Compute TTA loss."""
        prob = torch.sigmoid(logits)  # [B, 1]

        # Entropy minimization
        entropy = -(
            prob * torch.log(prob + 1e-8) +
            (1 - prob) * torch.log(1 - prob + 1e-8)
        )
        entropy_loss = entropy.mean()

        # Confidence regularization
        confidence_loss = -torch.abs(prob - 0.5).mean()

        # Total loss
        total_loss = (
            self.weights['entropy'] * entropy_loss +
            self.weights['confidence'] * confidence_loss
        )

        return {
            'total': total_loss,
            'entropy': entropy_loss.item(),
            'confidence': confidence_loss.item(),
            'mean_prob': prob.mean().item(),
        }


# ============================================================================
# Phase 9: TTA Inference Function
# ============================================================================

def inference_with_tta_v2(
    model: UnifiedTTANRAMv2,
    images: torch.Tensor,
    config: TTANRAMv2Config,
    return_debug: bool = False
) -> Dict:
    """
    Test-time adaptation inference with artifact-aware refinement.

    PROCESS:
    1. Initial forward (no TTA)
    2. TTA loop (K steps):
        - Forward with test_time=True
        - Compute self-supervised loss
        - Update artifact-conditional attention only
    3. Final forward (get final prediction)
    4. Update memory bank

    Args:
        model: UnifiedTTANRAMv2 model
        images: [B, 3, H, W]
        config: TTANRAMv2Config
        return_debug: Whether to return debug info

    Returns:
        dict with predictions, confidences, and optional debug info
    """
    device = config.device
    images = images.to(device)

    # ========================================
    # Phase 1: Initial Forward (No TTA)
    # ========================================
    model.eval()
    with torch.no_grad():
        logits_initial, _, debug_initial = model(images, test_time=True)
        pred_initial = torch.sigmoid(logits_initial)

    # ========================================
    # Phase 2: Enable Gradients for Artifact Attention Only
    # ========================================
    # Freeze everything except artifact-conditional attention
    for name, param in model.named_parameters():
        if 'artifact_attention' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # TTA loss function
    tta_loss_fn = TTALoss(weights=config.tta_loss_weights)

    # ========================================
    # Phase 3: TTA Loop
    # ========================================
    tta_history = []

    for step in range(config.tta_steps):
        # Forward with test_time=True
        logits, features, debug = model(images, test_time=True)

        # Compute self-supervised loss
        loss_dict = tta_loss_fn(logits)
        loss = loss_dict['total']

        # Backward (only artifact attention gets gradients)
        loss.backward()

        # Manual gradient update
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Gradient descent
                    param.data = param.data - config.tta_lr * param.grad
                    param.grad.zero_()

        # Record history
        prob = torch.sigmoid(logits).detach()
        tta_history.append({
            'step': step,
            'loss': loss.item(),
            'entropy': loss_dict['entropy'],
            'mean_prob': prob.mean().item(),
            'artifact_score_mean': debug['artifact_score_mean'],
            'noise_level_mean': debug['noise_level_mean'],
        })

    # ========================================
    # Phase 4: Final Prediction
    # ========================================
    model.eval()
    with torch.no_grad():
        logits_final, features_final, debug_final = model(images, test_time=True)
        pred_final = torch.sigmoid(logits_final)

    # ========================================
    # Phase 5: Update Memory Bank
    # ========================================
    if model.memory_bank is not None:
        with torch.no_grad():
            model.update_memory(features_final, logits_final)

    # ========================================
    # Phase 6: Disable Gradients
    # ========================================
    for param in model.parameters():
        param.requires_grad = False

    # ========================================
    # Prepare Results
    # ========================================
    results = {
        'predictions': pred_final.cpu(),
        'logits': logits_final.cpu(),
        'initial_predictions': pred_initial.cpu(),
        'improvement': (pred_final - pred_initial).mean().item(),
    }

    if return_debug:
        results['tta_history'] = tta_history
        results['debug_initial'] = debug_initial
        results['debug_final'] = debug_final

    return results


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters in model (total, trainable, frozen)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
    }


def print_model_info(model: UnifiedTTANRAMv2):
    """Print model architecture and parameter counts."""
    print("=" * 80)
    print("TTA-NRAM v2 (Artifact-Aware) Model Information")
    print("=" * 80)

    # Config
    print(f"\nConfiguration:")
    print(f"  Model: {model.config.model}")
    print(f"  Target Layer: {model.config.target_layer}")
    print(f"  TTA Steps: {model.config.tta_steps}")
    print(f"  Artifact Bands: {model.config.artifact_bands}")
    print(f"  Memory Bank: {'Enabled' if model.config.enable_memory_bank else 'Disabled'}")

    # Parameters
    params = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")

    # Components
    nram_params = count_parameters(model.nram)
    artifact_attn_params = count_parameters(model.nram.artifact_attention)
    print(f"\nComponent Breakdown:")
    print(f"  NRAM v2: {nram_params['total']:,} params")
    print(f"    ├─ Artifact Detector: 0 params (parameter-free)")
    print(f"    ├─ Noise Estimator: 0 params (parameter-free)")
    print(f"    └─ Artifact Attention: {artifact_attn_params['total']:,} params (trainable during TTA)")
    print(f"  Base Classifier: Using pre-trained (frozen)")

    print("=" * 80)


if __name__ == "__main__":
    print("TTA-NRAM v2 (Artifact-Aware) module loaded successfully!")
    print("\nUsage:")
    print("  from model.method.tta_nramv2 import UnifiedTTANRAMv2, TTANRAMv2Config, inference_with_tta_v2")
    print("\nKey features (NEW in v2):")
    print("  ✅ Frequency-domain artifact detection (parameter-free)")
    print("  ✅ Artifact-conditional dual-path attention")
    print("  ✅ Preserves deepfake signals under corruption")
    print("  ✅ Test-time adaptation in 5-10 steps")
    print("  ✅ Self-supervised (no labels needed)")
