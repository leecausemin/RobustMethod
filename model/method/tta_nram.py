"""
Test-Time Adaptive NRAM (Noise-Robust Attention Module) for Deepfake Detection

This module implements a test-time adaptation approach for robust deepfake detection
under various corruptions (Gaussian noise, JPEG compression, blur, etc.).

Key Features:
- Parameter-free noise estimation
- Dynamic channel-wise gating based on noise level
- Memory bank for robust statistics
- Self-supervised adaptation (no labels needed)
- Lightweight adapter classifier

Architecture:
    Base Model (frozen) → layer4 features
        ↓
    TTA-NRAM (adaptive gating)
        ↓
    Adapter Classifier (lightweight)
        ↓
    Final prediction

Author: [Your name]
Date: 2026-01-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TTANRAMConfig:
    """Configuration for TTA-NRAM"""

    # Model selection
    model: str = "LGrad"  # "LGrad" or "NPR"

    # Target layer (simplified to single layer)
    target_layer: str = None  # e.g., 'classifier.layer4' (auto-detected if None)

    # Channel attention
    reduction_ratio: int = 16  # SE-Net style reduction

    # Noise estimation
    noise_detection_method: str = "laplacian"  # "laplacian" or "variance"
    noise_normalize_factor: float = 100.0  # Scale variance to [0,1]

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
# Phase 1: Noise Estimator
# ============================================================================

class NoiseEstimator(nn.Module):
    """
    Parameter-free noise level estimation based on high-frequency analysis.

    Uses Laplacian filter to detect high-frequency components, which indicate
    the presence of noise. The variance of the filtered response is used as
    a proxy for noise level.

    Args:
        method: "laplacian" or "variance"
        normalize_factor: Scale to normalize variance to [0,1]

    Returns:
        noise_level: [B, 1] tensor in range [0, 1]
            - 0: Clean (no noise)
            - 1: Very noisy
    """

    def __init__(self, method: str = "laplacian", normalize_factor: float = 100.0):
        super().__init__()
        self.method = method
        self.normalize_factor = normalize_factor

        if method == "laplacian":
            # Laplacian kernel for edge/high-freq detection
            # [[0, -1, 0],
            #  [-1, 4, -1],
            #  [0, -1, 0]]
            laplacian_kernel = torch.tensor([
                [0., -1., 0.],
                [-1., 4., -1.],
                [0., -1., 0.]
            ]).view(1, 1, 3, 3) / 8.0  # Normalize

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
        """
        Laplacian-based high-frequency detection.

        High-frequency components (edges, noise) have large Laplacian response.
        We compute the variance of the Laplacian-filtered feature map as a
        measure of noise.
        """
        B, C, H, W = F.shape
        device = F.device

        # Apply Laplacian filter to each channel
        # We need to expand kernel to match channel count
        # Option 1: Apply same kernel to all channels (current)
        # Option 2: Learn different kernels per channel (more complex)

        kernel = self.laplacian_kernel.to(device)  # [1, 1, 3, 3]

        # Convolve each channel independently
        # Reshape F to [B*C, 1, H, W] to apply conv2d
        F_reshaped = F.view(B * C, 1, H, W)

        # Apply convolution
        import torch.nn.functional as F_func
        F_filtered = F_func.conv2d(F_reshaped, kernel, padding=1)  # [B*C, 1, H, W]

        # Reshape back to [B, C, H, W]
        F_filtered = F_filtered.view(B, C, H, W)

        # Compute variance of filtered response (high-freq energy)
        # Spatial variance per channel, then average across channels
        spatial_var = F_filtered.var(dim=[2, 3])  # [B, C]
        noise_var = spatial_var.mean(dim=1, keepdim=True)  # [B, 1]

        # Normalize to [0, 1] range
        noise_level = torch.clamp(noise_var / self.normalize_factor, 0.0, 1.0)

        return noise_level

    def _estimate_variance(self, F: torch.Tensor) -> torch.Tensor:
        """
        Simple spatial variance-based noise estimation.

        Assumption: Noisy images have higher spatial variance.
        """
        B, C, H, W = F.shape

        # Spatial variance per channel
        spatial_var = F.var(dim=[2, 3])  # [B, C]

        # Average across channels
        noise_var = spatial_var.mean(dim=1, keepdim=True)  # [B, 1]

        # Normalize to [0, 1]
        noise_level = torch.clamp(noise_var / self.normalize_factor, 0.0, 1.0)

        return noise_level


# ============================================================================
# Phase 2: Channel Attention
# ============================================================================

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) style channel attention.

    Learns to weight channels based on their importance for the task.
    This is a standard SE-Net block with:
    - Global average pooling (squeeze)
    - Two FC layers with bottleneck (excitation)
    - Sigmoid activation

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default: 16)

    Returns:
        attn_weights: [B, C, 1, 1] attention weights in [0, 1]
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


# ============================================================================
# Phase 3: Test-Time Adaptive NRAM (Simplified - No EMA)
# ============================================================================

class TestTimeAdaptiveNRAM(nn.Module):
    """
    Single TTA-NRAM layer for adaptive channel gating.

    SIMPLIFIED VERSION (per feedback):
    - ❌ NO EMA (removed complexity)
    - ✅ Use current batch statistics only
    - ✅ Memory bank handles accumulated statistics

    Process:
    1. Estimate noise level (parameter-free)
    2. Compute channel attention (learnable)
    3. Compute robustness score (variance-based)
    4. Adaptive gating = attention × (1 - noise) × robustness
    5. Apply gate with residual connection

    Args:
        channels: Number of input channels (e.g., 2048 for ResNet layer4)
        config: TTANRAMConfig
    """

    def __init__(self, channels: int, config: TTANRAMConfig):
        super().__init__()
        self.channels = channels
        self.config = config

        # Noise estimator (parameter-free)
        self.noise_estimator = NoiseEstimator(
            method=config.noise_detection_method,
            normalize_factor=config.noise_normalize_factor
        )

        # Channel attention (learnable)
        self.channel_attention = ChannelAttention(
            channels=channels,
            reduction=config.reduction_ratio
        )

    def forward(
        self,
        F: torch.Tensor,
        test_time: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Apply TTA-NRAM to feature map.

        Args:
            F: Feature map [B, C, H, W]
            test_time: Boolean (True during TTA, False during training)

        Returns:
            F_enhanced: Enhanced features [B, C, H, W]
            weights: Adaptive weights [B, C]
            noise_level: Estimated noise [B, 1]
            debug_info: Dict with diagnostic information
        """
        B, C, H, W = F.shape

        # ========================================
        # Phase 1: Noise Level Estimation
        # ========================================
        noise_level = self.noise_estimator(F)  # [B, 1]

        # ========================================
        # Phase 2: Feature Statistics (Current Batch Only)
        # ========================================
        # Compute variance per channel (for robustness score)
        feat_var = F.var(dim=[0, 2, 3])  # [C] - variance across batch and spatial dims

        # ========================================
        # Phase 3: Channel Attention
        # ========================================
        attn = self.channel_attention(F)  # [B, C, 1, 1]
        attn = attn.squeeze(-1).squeeze(-1)  # [B, C]

        # ========================================
        # Phase 4: Adaptive Gating (Noise-Conditioned)
        # ========================================
        # Interpretation:
        # - High noise → gate → 0 (suppress channels)
        # - Low noise → gate → 1 (keep channels)
        gate = 1.0 - noise_level  # [B, 1]
        gate = gate.expand(-1, C)  # [B, C] - broadcast to all channels

        # ========================================
        # Phase 5: Robustness Score (Variance-Based)
        # ========================================
        # Intuition: Channels with low variance are more stable/robust
        # We use exponential decay to convert variance to robustness score
        # Low variance → high robustness score → keep channel
        # High variance → low robustness score → suppress channel

        mean_var = feat_var.mean() + 1e-6  # Normalization factor
        robustness = torch.exp(-feat_var / mean_var)  # [C]
        robustness = robustness.unsqueeze(0).expand(B, -1)  # [B, C]

        # ========================================
        # Phase 6: Combine All Factors
        # ========================================
        # Final weight = attention × gate × robustness
        # - attention: learned channel importance
        # - gate: noise-based suppression
        # - robustness: variance-based stability
        weights = attn * gate * robustness  # [B, C]

        # ========================================
        # Phase 7: Apply Weighting
        # ========================================
        weights_4d = weights.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        F_gated = F * weights_4d  # [B, C, H, W]

        # ========================================
        # Phase 8: Residual Connection (Stability)
        # ========================================
        # Prevents over-suppression by keeping some of original features
        alpha = self.config.residual_weight
        F_enhanced = (1 - alpha) * F_gated + alpha * F  # [B, C, H, W]

        # ========================================
        # Debug Information
        # ========================================
        debug_info = {
            'noise_level_mean': noise_level.mean().item(),
            'attn_mean': attn.mean().item(),
            'gate_mean': gate.mean().item(),
            'robustness_mean': robustness.mean().item(),
            'weights_mean': weights.mean().item(),
            'weights_std': weights.std().item(),
            'feat_var_mean': feat_var.mean().item(),
        }

        return F_enhanced, weights, noise_level, debug_info


# ============================================================================
# Phase 4: Memory Bank
# ============================================================================

class MemoryBank(nn.Module):
    """
    Confidence-weighted memory bank for robust statistics.

    KEY DESIGN (per feedback):
    - Stores only HIGH-CONFIDENCE samples (>threshold)
    - Uses FIFO (First-In-First-Out) queue
    - EMA smoothing when updating existing slots
    - Confidence-weighted aggregation

    Purpose:
    - Prevent model collapse from noisy/outlier samples
    - Provide robust accumulated statistics over test stream
    - Enable continual adaptation without forgetting

    Args:
        num_channels: Channel dimension (e.g., 2048)
        memory_size: Queue size (default: 100)
        confidence_threshold: Only store samples with confidence > threshold
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
            torch.tensor(0, dtype=torch.long)  # Number of filled slots
        )

    def update(
        self,
        features: torch.Tensor,
        confidence: torch.Tensor,
        ema_decay: float = 0.99
    ):
        """
        Update memory with high-confidence samples.

        TIMING (per feedback): Called AFTER TTA loop finishes and final
        prediction is obtained (so confidence is available).

        Args:
            features: [B, C] - Channel-wise features (global avg pooled)
            confidence: [B] - Prediction confidence in [0, 1]
            ema_decay: Smoothing factor for updating existing slots
        """
        B, C = features.shape
        assert C == self.num_channels, f"Expected {self.num_channels} channels, got {C}"

        # Filter high-confidence samples only
        high_conf_mask = confidence > self.confidence_threshold

        if high_conf_mask.sum() == 0:
            # No high-confidence samples, skip update
            return

        high_conf_features = features[high_conf_mask]  # [K, C] where K <= B
        high_conf_scores = confidence[high_conf_mask]  # [K]

        # Update memory (FIFO with EMA smoothing)
        for feat, conf in zip(high_conf_features, high_conf_scores):
            ptr = self.memory_pointer % self.memory_size

            if self.memory_filled > ptr:
                # Slot already filled, use EMA to blend
                self.memory_features[ptr] = (
                    ema_decay * self.memory_features[ptr] +
                    (1 - ema_decay) * feat
                )
                self.memory_confidences[ptr] = (
                    ema_decay * self.memory_confidences[ptr] +
                    (1 - ema_decay) * conf
                )
            else:
                # First time filling this slot
                self.memory_features[ptr] = feat
                self.memory_confidences[ptr] = conf
                self.memory_filled += 1

            self.memory_pointer += 1

    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """
        Get confidence-weighted statistics from memory.

        Returns:
            dict with 'mean' [C] and 'std' [C]
        """
        if self.memory_filled == 0:
            # No data in memory yet, return default (zeros/ones)
            return {
                'mean': torch.zeros(self.num_channels, device=self.memory_features.device),
                'std': torch.ones(self.num_channels, device=self.memory_features.device),
                'num_samples': 0
            }

        # Use only filled slots
        valid_features = self.memory_features[:self.memory_filled]  # [N, C]
        valid_confidences = self.memory_confidences[:self.memory_filled]  # [N]

        # Confidence-weighted mean
        total_conf = valid_confidences.sum() + 1e-8
        weighted_mean = (
            valid_features * valid_confidences.unsqueeze(-1)
        ).sum(dim=0) / total_conf  # [C]

        # Confidence-weighted standard deviation
        diff = valid_features - weighted_mean.unsqueeze(0)  # [N, C]
        weighted_var = (
            (diff ** 2) * valid_confidences.unsqueeze(-1)
        ).sum(dim=0) / total_conf  # [C]
        weighted_std = torch.sqrt(weighted_var + 1e-8)  # [C]

        return {
            'mean': weighted_mean,
            'std': weighted_std,
            'num_samples': self.memory_filled.item()
        }

    def reset(self):
        """
        Clear memory (e.g., when switching to new test set or corruption type).
        """
        self.memory_features.zero_()
        self.memory_confidences.zero_()
        self.memory_pointer.zero_()
        self.memory_filled.zero_()


# ============================================================================
# Phase 5: Unified TTA-NRAM Wrapper (No Adapter Needed!)
# ============================================================================

class FeatureExtractor:
    """
    Hook-based feature extractor for intermediate layers.

    Simple utility to extract features from a specific layer during forward pass.
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


class UnifiedTTANRAM(nn.Module):
    """
    Unified TTA-NRAM wrapper for LGrad and NPR models.

    SIMPLIFIED ARCHITECTURE (NO ADAPTER NEEDED):

        Input Image [B, 3, H, W]
            ↓
        Base Model (frozen) → extract layer4 features
            ↓
        layer4 features [B, 2048, 7, 7]
            ↓
        TestTimeAdaptiveNRAM (adaptive gating)
            ↓
        Enhanced features [B, 2048, 7, 7]
            ↓
        Base Classifier (avgpool + fc) - Pre-trained!
            ↓
        Logits [B, 1]

    KEY DESIGN:
    - ✅ Single target layer (layer4 only)
    - ✅ NO adapter training needed - uses pre-trained classifier
    - ✅ NRAM refines features in same feature space
    - ✅ Clear separation: Base → NRAM → Base Classifier
    - ✅ Memory bank for robust statistics

    Args:
        base_model: Pre-trained LGrad or NPR model (will be frozen)
        config: TTANRAMConfig
    """

    def __init__(self, base_model: nn.Module, config: TTANRAMConfig):
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

        # TTA-NRAM module (single layer)
        self.nram = TestTimeAdaptiveNRAM(channels, config).to(config.device)

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
        Forward pass through TTA-NRAM.

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
        # Step 2: Apply TTA-NRAM
        # ========================================
        feat_enhanced, weights, noise_level, debug_info = self.nram(
            features,
            test_time=test_time
        )

        # ========================================
        # Step 3: Use Base Model's Classifier (avgpool + fc)
        # ========================================
        # For LGrad: base_model.classifier is ResNet50
        # For NPR: base_model.resnet is ResNet50
        # We apply avgpool + fc from the base classifier

        if self.config.model == "LGrad":
            classifier = self.base_model.classifier
        elif self.config.model == "NPR":
            classifier = self.base_model.resnet
        else:
            raise ValueError(f"Unknown model: {self.config.model}")

        # Apply avgpool
        feat_pooled = classifier.avgpool(feat_enhanced)  # [B, C, 1, 1]
        feat_pooled = torch.flatten(feat_pooled, 1)     # [B, C]

        # Apply fc (final classifier)
        logits = classifier.fc(feat_pooled)  # [B, 1]

        # ========================================
        # Return
        # ========================================
        if test_time:
            return logits, feat_pooled, debug_info
        else:
            return logits, None, None

    def update_memory(self, features: torch.Tensor, logits: torch.Tensor):
        """
        Update memory bank with features and confidence.

        TIMING (per feedback): Called AFTER TTA loop and final prediction.

        Args:
            features: [B, C] - Enhanced features (global pooled)
            logits: [B, 1] - Final predictions
        """
        if self.memory_bank is None:
            warnings.warn("Memory bank is disabled, skipping update")
            return

        # Convert logits to confidence [0, 1]
        # For binary classification: confidence = max(prob, 1-prob)
        prob = torch.sigmoid(logits).squeeze(-1)  # [B]
        confidence = torch.maximum(prob, 1 - prob)  # [B]

        # Update memory
        self.memory_bank.update(features, confidence)

    def reset_memory(self):
        """Reset memory bank (e.g., when switching test sets)."""
        if self.memory_bank is not None:
            self.memory_bank.reset()

    def _get_default_target_layer(self) -> str:
        """Auto-detect target layer based on model type."""
        if self.config.model == "LGrad":
            return 'classifier.layer4'
        elif self.config.model == "NPR":
            return 'resnet.layer4'
        else:
            raise ValueError(f"Unknown model type: {self.config.model}")

    def _get_layer_channels(self, layer_name: str) -> int:
        """
        Get number of output channels for a layer.

        Uses a dummy forward pass to determine the channel count.
        """
        device = self.config.device
        dummy_input = torch.zeros(1, 3, 224, 224).to(device)

        with torch.no_grad():
            _ = self.base_model(dummy_input)

        feature = self.feature_extractor.features[layer_name]
        return feature.shape[1]  # C from [B, C, H, W]


# ============================================================================
# Phase 7: TTA Loss and Inference Loop
# ============================================================================

class TTALoss(nn.Module):
    """
    Self-supervised loss for test-time adaptation.

    NO LABELS NEEDED! Uses prediction confidence and entropy.

    Components:
    1. Entropy Minimization: Push predictions to be confident (low entropy)
    2. Confidence Regularization: Push away from uncertain 0.5

    Args:
        weights: Dict with loss component weights
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.weights = weights or {'entropy': 1.0, 'confidence': 0.1}

    def forward(self, logits: torch.Tensor) -> Dict[str, float]:
        """
        Compute TTA loss.

        Args:
            logits: [B, 1] - Model predictions (before sigmoid)

        Returns:
            dict with 'total' loss and individual components
        """
        prob = torch.sigmoid(logits)  # [B, 1]

        # ========================================
        # Loss 1: Entropy Minimization
        # ========================================
        # Entropy = -[p*log(p) + (1-p)*log(1-p)]
        # Low entropy → confident prediction (good)
        # High entropy → uncertain prediction (bad)
        entropy = -(
            prob * torch.log(prob + 1e-8) +
            (1 - prob) * torch.log(1 - prob + 1e-8)
        )  # [B, 1]
        entropy_loss = entropy.mean()

        # ========================================
        # Loss 2: Confidence Regularization
        # ========================================
        # Push predictions away from 0.5 (uncertain)
        # |p - 0.5| → maximize → -|p - 0.5| to minimize
        confidence_loss = -torch.abs(prob - 0.5).mean()

        # ========================================
        # Total Loss
        # ========================================
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


def inference_with_tta(
    model: UnifiedTTANRAM,
    images: torch.Tensor,
    config: TTANRAMConfig,
    return_debug: bool = False
) -> Dict:
    """
    Test-time adaptation inference with iterative refinement.

    PROCESS (per feedback):
    1. Initial forward (no TTA)
    2. TTA loop (5 steps):
        - Forward with test_time=True
        - Compute self-supervised loss
        - Update NRAM parameters only
    3. Final forward (get final prediction)
    4. Update memory bank ← CLEAR TIMING

    Args:
        model: UnifiedTTANRAM model
        images: [B, 3, H, W]
        config: TTANRAMConfig
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
        logits_initial, _, _ = model(images, test_time=False)
        pred_initial = torch.sigmoid(logits_initial)

    # ========================================
    # Phase 2: Enable Gradients for NRAM Only
    # ========================================
    # Freeze everything except NRAM
    for name, param in model.named_parameters():
        if 'nram' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # TTA loss function
    tta_loss_fn = TTALoss(weights=config.tta_loss_weights)

    # ========================================
    # Phase 3: TTA Loop (5 Steps)
    # ========================================
    tta_history = []

    for step in range(config.tta_steps):
        # Forward with test_time=True
        logits, features, debug = model(images, test_time=True)

        # Compute self-supervised loss
        loss_dict = tta_loss_fn(logits)
        loss = loss_dict['total']

        # Backward (only NRAM parameters get gradients)
        loss.backward()

        # Manual gradient update (no optimizer)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Gradient descent: θ = θ - lr * ∇θ
                    param.data = param.data - config.tta_lr * param.grad
                    param.grad.zero_()

        # Record history
        prob = torch.sigmoid(logits).detach()
        tta_history.append({
            'step': step,
            'loss': loss.item(),
            'entropy': loss_dict['entropy'],
            'mean_prob': prob.mean().item(),
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
    # CRITICAL TIMING (per feedback): After TTA loop, after final prediction
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
        'predictions': pred_final.cpu(),  # [B, 1]
        'logits': logits_final.cpu(),     # [B, 1]
        'initial_predictions': pred_initial.cpu(),
        'improvement': (pred_final - pred_initial).mean().item(),
    }

    if return_debug:
        results['tta_history'] = tta_history
        results['debug_initial'] = None
        results['debug_final'] = debug_final

    return results


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in model (total, trainable, frozen).

    Useful for verifying that base model is frozen and only NRAM is trainable.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
    }


def print_model_info(model: UnifiedTTANRAM):
    """Print model architecture and parameter counts."""
    print("=" * 80)
    print("TTA-NRAM Model Information")
    print("=" * 80)

    # Config
    print(f"\nConfiguration:")
    print(f"  Model: {model.config.model}")
    print(f"  Target Layer: {model.config.target_layer}")
    print(f"  TTA Steps: {model.config.tta_steps}")
    print(f"  Memory Bank: {'Enabled' if model.config.enable_memory_bank else 'Disabled'}")

    # Parameters
    params = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")

    # Components
    nram_params = count_parameters(model.nram)
    print(f"\nComponent Breakdown:")
    print(f"  NRAM: {nram_params['total']:,} params (trainable during TTA)")
    print(f"  Base Classifier: Using pre-trained (frozen)")

    print("=" * 80)


if __name__ == "__main__":
    print("TTA-NRAM module loaded successfully!")
    print("Use 'from model.method.tta_nram import UnifiedTTANRAM, TTANRAMConfig'")
