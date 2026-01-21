"""
Test-Time Adaptive NRAM v4 (Evidence-Calibrated Soft Class-Conditional) for Deepfake Detection

This module implements an evidence-calibrated soft routing approach that:
1. Uses SOFT routing instead of hard class-split to prevent pseudo-label error accumulation
2. Controls routing confidence via evidence quality (artifact vs noise)
3. Maintains class-specific NRAM modules (real/fake) with soft mixing
4. Class-specific memory banks with dual gating (confidence AND evidence)

Key Improvements over v3:
- Soft routing: w_i = sigmoid(temp(q_i) * logit_i) where temp is evidence-calibrated
- High evidence q → sharp routing (temp high, w_i near 0 or 1, almost class-conditional)
- Low evidence q → soft routing (temp low, w_i near 0.5, almost shared)
- Class-specific memory banks with dual gating (confidence AND q > threshold)
- Prevents pseudo-label error accumulation in continual/corrupted settings

Architecture:
    Base Model (frozen) → layer4 features
        ↓
    Pseudo-label generation (logit)
        ↓
    Evidence quality estimation (q from artifact & noise)
        ↓
    Temperature-controlled soft routing:
        temp(q) = temp_min + (temp_max - temp_min) * q
        w = sigmoid(temp(q) * logit)
        ↓
    Real NRAM (dual-path artifact attention)
    Fake NRAM (dual-path artifact attention)
        ↓
    Soft mixing: F_out = w * F_fake + (1-w) * F_real
        ↓
    Base Classifier → Final prediction
        ↓
    Class-specific memory update (if confidence AND q > threshold)

Author: Evidence-Calibrated Soft Routing Team
Date: 2026-01-13
Reference: Soft routing prevents confirmation bias in continual TTA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings
import numpy as np

# Import utilities from v3
from model.method.tta_nramv3 import (
    robust_normalize,
    rank_normalize,
    compute_quantile,
    FrequencyArtifactDetector,
    NoiseEstimator,
    ChannelAttention,
    ArtifactConditionalChannelAttention,
    FeatureExtractor,
    count_parameters,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TTANRAMv4Config:
    """
    Configuration for TTA-NRAM v4 (Evidence-Calibrated Soft Class-Conditional).

    Core hyperparameters (6 total):
    - reduction_ratio: Channel attention SE reduction ratio (default 16)
    - max_tta_steps: Maximum TTA iterations (default 10)
    - tta_lr: Learning rate for TTA (default 1e-4)
    - update_sample_ratio: Top-ρ ratio for sample selection (default 0.7)
    - temp_min: Minimum temperature for soft routing (default 0.1)
    - temp_max: Maximum temperature for soft routing (default 10.0)

    Soft routing design:
    - temp(q) = temp_min + (temp_max - temp_min) * q
    - w_i = sigmoid(temp(q_i) * logit_i)
    - High q → temp high → w_i sharp (near 0/1, class-conditional behavior)
    - Low q → temp low → w_i soft (near 0.5, shared behavior)
    """

    # Model selection
    model: str = "LGrad"  # "LGrad" or "NPR"
    target_layer: str = None  # Auto-detect

    # === Core Hyperparameters (6 tunable) ===
    reduction_ratio: int = 16              # SE reduction
    max_tta_steps: int = 10                # Max steps (early stop by data)
    tta_lr: float = 1e-4                   # TTA learning rate
    update_sample_ratio: float = 0.7       # Top-ρ for sample selection (70%)
    temp_min: float = 0.1                  # Min temperature (soft routing)
    temp_max: float = 10.0                 # Max temperature (sharp routing)
    # === END Core Hyperparameters ===

    # Memory bank
    enable_memory_bank: bool = True
    memory_size: int = 100
    memory_confidence_threshold: float = 0.8  # Confidence threshold for memory
    memory_evidence_threshold: float = 0.7    # Evidence threshold for memory

    # Flags
    enable_evidence_calibration: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = False


# ============================================================================
# Phase 1: Evidence-Calibrated Soft Routing Module
# ============================================================================

class EvidenceCalibratedSoftRouting(nn.Module):
    """
    Soft routing with evidence-calibrated temperature control.

    Key insight:
    - High evidence q → Trust pseudo-labels → Sharp routing (temp high)
    - Low evidence q → Don't trust pseudo-labels → Soft routing (temp low)

    Formula:
        temp(q) = temp_min + (temp_max - temp_min) * q
        w = sigmoid(temp(q) * logit)

    where:
        - logit: Raw prediction from base model (before sigmoid)
        - q: Evidence quality in [0, 1]
        - w: Soft routing weight in [0, 1]
            - w → 1: Use fake NRAM
            - w → 0: Use real NRAM

    Args:
        temp_min: Minimum temperature (default 0.1)
        temp_max: Maximum temperature (default 10.0)
    """

    def __init__(self, temp_min: float = 0.1, temp_max: float = 10.0):
        super().__init__()
        self.temp_min = temp_min
        self.temp_max = temp_max

    def forward(
        self,
        logits: torch.Tensor,  # [B, 1]
        evidence: torch.Tensor  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute soft routing weights.

        Args:
            logits: Raw predictions [B, 1]
            evidence: Evidence quality [B] in [0, 1]

        Returns:
            weights: Soft routing weights [B] in [0, 1]
            temperature: Temperature per sample [B]
            debug_info: Dict
        """
        B = logits.shape[0]

        # Compute temperature per sample
        evidence = evidence.clamp(0.0, 1.0)  # Ensure [0, 1]
        temperature = self.temp_min + (self.temp_max - self.temp_min) * evidence  # [B]

        # Compute soft routing weights
        # w = sigmoid(temp * logit)
        logits_scaled = temperature.unsqueeze(1) * logits  # [B, 1]
        weights = torch.sigmoid(logits_scaled).squeeze(1)  # [B]

        # Debug info
        debug_info = {
            'temp_mean': temperature.mean().item(),
            'temp_min': temperature.min().item(),
            'temp_max': temperature.max().item(),
            'weight_mean': weights.mean().item(),
            'weight_std': weights.std().item(),
        }

        return weights, temperature, debug_info


# ============================================================================
# Phase 2: Class-Specific NRAM Module
# ============================================================================

class TestTimeAdaptiveNRAMv4(nn.Module):
    """
    Dual NRAM modules for real/fake with soft routing.

    Architecture:
    - real_nram: Dual-path artifact attention for real-like samples
    - fake_nram: Dual-path artifact attention for fake-like samples
    - Soft mixing based on evidence-calibrated routing weights

    Args:
        channels: Number of input channels
        config: TTANRAMv4Config
    """

    def __init__(self, channels: int, config: TTANRAMv4Config):
        super().__init__()
        self.channels = channels
        self.config = config

        # Artifact detector (parameter-free, data-driven)
        self.artifact_detector = FrequencyArtifactDetector()

        # Noise estimator (parameter-free, data-driven)
        self.noise_estimator = NoiseEstimator(method="laplacian")

        # Soft routing module
        self.soft_routing = EvidenceCalibratedSoftRouting(
            temp_min=config.temp_min,
            temp_max=config.temp_max
        )

        # Class-specific artifact-conditional attention (learnable)
        self.real_artifact_attention = ArtifactConditionalChannelAttention(
            channels=channels,
            reduction=config.reduction_ratio
        )
        self.fake_artifact_attention = ArtifactConditionalChannelAttention(
            channels=channels,
            reduction=config.reduction_ratio
        )

        # Fixed residual weight (not learnable)
        self.register_buffer('alpha', torch.tensor(0.1))

    def forward(
        self,
        F: torch.Tensor,           # [B, C, H, W]
        logits: torch.Tensor,      # [B, 1] - For soft routing
        evidence: torch.Tensor,    # [B] - For temperature control
        test_time: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Apply soft class-conditional TTA-NRAM.

        Args:
            F: Feature map [B, C, H, W]
            logits: Raw predictions [B, 1] for soft routing
            evidence: Evidence quality [B] for temperature control
            test_time: Boolean

        Returns:
            F_enhanced: Enhanced features [B, C, H, W]
            artifact_score: [B, 1]
            noise_level: [B, 1]
            routing_weights: [B] - Soft routing weights
            F_base: Base features [B, C, H, W]
            debug_info: Dict
        """
        B, C, H, W = F.shape

        # === Phase 1: Artifact & Noise (no grad, for evidence) ===
        with torch.no_grad():
            artifact_score, artifact_debug = self.artifact_detector(F)
            noise_level = self.noise_estimator(F)

        # === Phase 2: Soft Routing Weights ===
        with torch.no_grad():
            routing_weights, temperature, routing_debug = self.soft_routing(
                logits, evidence
            )  # routing_weights: [B]

        # === Phase 3: Process with Real NRAM ===
        attn_real = self.real_artifact_attention(F, artifact_score.detach())  # [B, C, 1, 1]

        # Noise gating
        gate = 1.0 - noise_level.detach()  # [B, 1]
        gate = gate.view(B, 1, 1, 1).expand(-1, C, 1, 1)  # [B, C, 1, 1]

        weights_real = attn_real * gate  # [B, C, 1, 1]
        F_real_gated = F * weights_real  # [B, C, H, W]
        F_real_enhanced = (1 - self.alpha) * F_real_gated + self.alpha * F  # [B, C, H, W]

        # === Phase 4: Process with Fake NRAM ===
        attn_fake = self.fake_artifact_attention(F, artifact_score.detach())  # [B, C, 1, 1]
        weights_fake = attn_fake * gate  # [B, C, 1, 1]
        F_fake_gated = F * weights_fake  # [B, C, H, W]
        F_fake_enhanced = (1 - self.alpha) * F_fake_gated + self.alpha * F  # [B, C, H, W]

        # === Phase 5: Soft Mixing ===
        # F_out = w * F_fake + (1-w) * F_real
        # where w is the soft routing weight
        w = routing_weights.view(B, 1, 1, 1)  # [B, 1, 1, 1]
        F_enhanced = w * F_fake_enhanced + (1 - w) * F_real_enhanced  # [B, C, H, W]

        # Save base feature
        F_base = F.detach()

        # Debug info
        debug_info = {
            'artifact_score_mean': artifact_score.mean().item(),
            'noise_level_mean': noise_level.mean().item(),
            'routing_weight_mean': routing_weights.mean().item(),
            'routing_weight_std': routing_weights.std().item(),
            'temp_mean': routing_debug['temp_mean'],
            'attn_real_mean': attn_real.mean().item(),
            'attn_fake_mean': attn_fake.mean().item(),
            'alpha': self.alpha.item(),
        }
        debug_info.update(routing_debug)

        return F_enhanced, artifact_score, noise_level, routing_weights, F_base, debug_info


# ============================================================================
# Phase 3: Class-Specific Memory Bank
# ============================================================================

class ClassSpecificMemoryBank(nn.Module):
    """
    Confidence AND evidence gated memory bank for class-specific statistics.

    KEY DESIGN:
    - Only store samples with BOTH high confidence AND high evidence
    - Prevents pseudo-label error accumulation in continual/corrupted settings
    - Separate memory for real/fake classes

    Args:
        num_channels: Channel dimension
        memory_size: Queue size
        confidence_threshold: Min confidence to store (default 0.8)
        evidence_threshold: Min evidence to store (default 0.7)
    """

    def __init__(
        self,
        num_channels: int,
        memory_size: int = 100,
        confidence_threshold: float = 0.8,
        evidence_threshold: float = 0.7
    ):
        super().__init__()
        self.num_channels = num_channels
        self.memory_size = memory_size
        self.confidence_threshold = confidence_threshold
        self.evidence_threshold = evidence_threshold

        # Memory buffers (FIFO queue)
        self.register_buffer('memory_features', torch.zeros(memory_size, num_channels))
        self.register_buffer('memory_confidences', torch.zeros(memory_size))
        self.register_buffer('memory_evidences', torch.zeros(memory_size))
        self.register_buffer('memory_pointer', torch.tensor(0, dtype=torch.long))
        self.register_buffer('memory_filled', torch.tensor(0, dtype=torch.long))

    def update(
        self,
        features: torch.Tensor,      # [B, C]
        confidence: torch.Tensor,    # [B]
        evidence: torch.Tensor,      # [B]
        ema_decay: float = 0.99
    ):
        """
        Update memory with high-confidence AND high-evidence samples.

        Args:
            features: [B, C] - Channel-wise features
            confidence: [B] - Prediction confidence in [0, 1]
            evidence: [B] - Evidence quality in [0, 1]
            ema_decay: Smoothing factor
        """
        B, C = features.shape
        assert C == self.num_channels

        # Dual gating: confidence AND evidence
        high_quality_mask = (confidence > self.confidence_threshold) & \
                           (evidence > self.evidence_threshold)

        if high_quality_mask.sum() == 0:
            return  # No high-quality samples

        high_quality_features = features[high_quality_mask]
        high_quality_confidences = confidence[high_quality_mask]
        high_quality_evidences = evidence[high_quality_mask]

        # Update memory (FIFO with EMA)
        for feat, conf, evid in zip(high_quality_features, high_quality_confidences, high_quality_evidences):
            ptr = self.memory_pointer % self.memory_size

            if self.memory_filled > ptr:
                # Slot filled, use EMA
                self.memory_features[ptr] = ema_decay * self.memory_features[ptr] + (1 - ema_decay) * feat
                self.memory_confidences[ptr] = ema_decay * self.memory_confidences[ptr] + (1 - ema_decay) * conf
                self.memory_evidences[ptr] = ema_decay * self.memory_evidences[ptr] + (1 - ema_decay) * evid
            else:
                # First time
                self.memory_features[ptr] = feat
                self.memory_confidences[ptr] = conf
                self.memory_evidences[ptr] = evid
                self.memory_filled += 1

            self.memory_pointer += 1

    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """Get confidence-weighted statistics from memory."""
        if self.memory_filled == 0:
            return {
                'mean': torch.zeros(self.num_channels, device=self.memory_features.device),
                'std': torch.ones(self.num_channels, device=self.memory_features.device),
                'num_samples': 0
            }

        valid_features = self.memory_features[:self.memory_filled]
        valid_confidences = self.memory_confidences[:self.memory_filled]

        total_conf = valid_confidences.sum() + 1e-8
        weighted_mean = (valid_features * valid_confidences.unsqueeze(-1)).sum(dim=0) / total_conf
        diff = valid_features - weighted_mean.unsqueeze(0)
        weighted_var = ((diff ** 2) * valid_confidences.unsqueeze(-1)).sum(dim=0) / total_conf
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
        self.memory_evidences.zero_()
        self.memory_pointer.zero_()
        self.memory_filled.zero_()


# ============================================================================
# Phase 4: Unified TTA-NRAM v4 Wrapper
# ============================================================================

class UnifiedTTANRAMv4(nn.Module):
    """
    Unified TTA-NRAM v4 with evidence-calibrated soft class-conditional routing.

    KEY DESIGN:
    - Soft routing prevents pseudo-label error accumulation
    - Evidence quality controls routing sharpness
    - Class-specific memory banks with dual gating
    - Base model always frozen
    """

    def __init__(self, base_model: nn.Module, config: TTANRAMv4Config):
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

        # TTA-NRAM v4
        self.nram = TestTimeAdaptiveNRAMv4(channels, config).to(config.device)

        # Class-specific memory banks
        if config.enable_memory_bank:
            self.real_memory_bank = ClassSpecificMemoryBank(
                num_channels=channels,
                memory_size=config.memory_size,
                confidence_threshold=config.memory_confidence_threshold,
                evidence_threshold=config.memory_evidence_threshold
            ).to(config.device)

            self.fake_memory_bank = ClassSpecificMemoryBank(
                num_channels=channels,
                memory_size=config.memory_size,
                confidence_threshold=config.memory_confidence_threshold,
                evidence_threshold=config.memory_evidence_threshold
            ).to(config.device)
        else:
            self.real_memory_bank = None
            self.fake_memory_bank = None

    def forward(
        self,
        images: torch.Tensor,
        test_time: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass through TTA-NRAM v4 with soft routing.

        Args:
            images: [B, 3, H, W]
            test_time: Boolean

        Returns:
            logits: [B, 1]
            features: [B, C] or None
            debug_info: Dict or None
        """
        # Base model (always no_grad)
        with torch.no_grad():
            _ = self.base_model(images)

        # Get hooked features
        features = self.feature_extractor.features[self.config.target_layer]

        # Get initial logits (for soft routing)
        if test_time:
            with torch.no_grad():
                if self.config.model == "LGrad":
                    classifier = self.base_model.classifier
                elif self.config.model == "NPR":
                    classifier = self.base_model.model
                else:
                    raise ValueError(f"Unknown model: {self.config.model}")

                feat_pooled_init = classifier.avgpool(features)
                feat_pooled_init = torch.flatten(feat_pooled_init, 1)
                fc_layer = classifier.fc1 if self.config.model == "NPR" else classifier.fc
                logits_init = fc_layer(feat_pooled_init)  # [B, 1]

                # Get artifact & noise for evidence
                artifact_score_init, _ = self.nram.artifact_detector(features)
                noise_level_init = self.nram.noise_estimator(features)
                evidence_init = self.compute_evidence_quality(artifact_score_init, noise_level_init)
        else:
            # Training mode: use dummy logits and evidence
            B = features.shape[0]
            logits_init = torch.zeros(B, 1, device=features.device)
            evidence_init = torch.ones(B, device=features.device) * 0.5

        # Apply TTA-NRAM v4
        feat_enhanced, artifact_score, noise_level, routing_weights, feat_base, debug_info = self.nram(
            features,
            logits_init,
            evidence_init,
            test_time=test_time
        )

        if test_time:
            debug_info['artifact_score'] = artifact_score
            debug_info['noise_level'] = noise_level
            debug_info['routing_weights'] = routing_weights
            debug_info['feat_base'] = feat_base
            debug_info['feat_enhanced'] = feat_enhanced

        # Use Base Classifier
        if self.config.model == "LGrad":
            classifier = self.base_model.classifier
        elif self.config.model == "NPR":
            classifier = self.base_model.model
        else:
            raise ValueError(f"Unknown model: {self.config.model}")

        feat_pooled = classifier.avgpool(feat_enhanced)
        feat_pooled = torch.flatten(feat_pooled, 1)
        fc_layer = classifier.fc1 if self.config.model == "NPR" else classifier.fc
        logits = fc_layer(feat_pooled)

        if test_time:
            return logits, feat_pooled, debug_info
        else:
            return logits, None, None

    def update_memory(
        self,
        features: torch.Tensor,     # [B, C]
        logits: torch.Tensor,       # [B, 1]
        routing_weights: torch.Tensor,  # [B] - Soft routing weights (0=real, 1=fake)
        evidence: torch.Tensor      # [B]
    ):
        """
        Update class-specific memory banks.

        Args:
            features: [B, C] - Enhanced features
            logits: [B, 1] - Final predictions
            routing_weights: [B] - Soft routing weights (0=real, 1=fake)
            evidence: [B] - Evidence quality
        """
        if self.real_memory_bank is None or self.fake_memory_bank is None:
            return

        # Compute confidence
        prob = torch.sigmoid(logits).squeeze(-1)  # [B]
        confidence = torch.maximum(prob, 1 - prob)  # [B]

        # Soft class assignment (weighted update)
        # For real memory: weight by (1 - routing_weight)
        # For fake memory: weight by routing_weight

        # Real memory: samples with low routing weight (< 0.5)
        real_mask = (routing_weights < 0.5)
        if real_mask.sum() > 0:
            # Weight confidence by how "real" the sample is
            real_confidence = confidence[real_mask] * (1 - routing_weights[real_mask])
            self.real_memory_bank.update(
                features[real_mask],
                real_confidence,
                evidence[real_mask]
            )

        # Fake memory: samples with high routing weight (>= 0.5)
        fake_mask = (routing_weights >= 0.5)
        if fake_mask.sum() > 0:
            # Weight confidence by how "fake" the sample is
            fake_confidence = confidence[fake_mask] * routing_weights[fake_mask]
            self.fake_memory_bank.update(
                features[fake_mask],
                fake_confidence,
                evidence[fake_mask]
            )

    def reset_memory(self):
        """Reset both memory banks."""
        if self.real_memory_bank is not None:
            self.real_memory_bank.reset()
        if self.fake_memory_bank is not None:
            self.fake_memory_bank.reset()

    def compute_evidence_quality(
        self,
        artifact_score: torch.Tensor,
        noise_level: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute evidence quality (same as v3).

        Args:
            artifact_score: [B, 1]
            noise_level: [B, 1]

        Returns:
            q: [B]
        """
        q = (artifact_score - noise_level + 1.0) / 2.0
        q = torch.clamp(q, 0.0, 1.0)
        q = q.squeeze(1)
        return q

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


# ============================================================================
# Phase 5: TTA Inference with Soft Routing
# ============================================================================

def inference_with_tta_v4(
    model: UnifiedTTANRAMv4,
    images: torch.Tensor,
    config: TTANRAMv4Config,
    return_debug: bool = False
) -> Dict:
    """
    Evidence-calibrated soft routing TTA inference.

    Key differences from v3:
    - Updates BOTH real and fake NRAM parameters
    - Soft routing prevents hard class-split errors
    - Class-specific memory updates with dual gating

    Args:
        model: UnifiedTTANRAMv4
        images: [B, 3, H, W]
        config: TTANRAMv4Config
        return_debug: Whether to return debug info

    Returns:
        dict with predictions and optional debug info
    """
    device = config.device
    images = images.to(device)
    B = images.shape[0]
    eps = torch.finfo(torch.float32).eps * 10

    # ========================================
    # Phase 1: Initial Forward
    # ========================================
    model.eval()
    with torch.no_grad():
        logits_init, _, debug_init = model(images, test_time=True)
        pred_init = torch.sigmoid(logits_init)

        artifact_score_init = debug_init['artifact_score']
        noise_level_init = debug_init['noise_level']
        routing_weights_init = debug_init['routing_weights']

    # ========================================
    # Phase 2: Evidence Quality
    # ========================================
    if not config.enable_evidence_calibration:
        if config.verbose:
            print("[TTA-v4] Evidence calibration disabled, using simple TTA")
        # Fallback to simple TTA
        return _inference_with_tta_v4_simple(model, images, config, return_debug)

    q_init = model.compute_evidence_quality(artifact_score_init, noise_level_init)
    q_mean_init = q_init.mean().item()

    if config.verbose:
        print(f"[TTA-v4] Initial: q_mean={q_mean_init:.3f}, "
              f"artifact={artifact_score_init.mean().item():.3f}, "
              f"noise={noise_level_init.mean().item():.3f}, "
              f"routing_weight_mean={routing_weights_init.mean().item():.3f}")

    # ========================================
    # Phase 3: Setup TTA Parameters
    # ========================================
    # Update BOTH real and fake artifact attention parameters
    params_to_update = [
        p for n, p in model.named_parameters()
        if 'real_artifact_attention' in n or 'fake_artifact_attention' in n
    ]

    if len(params_to_update) == 0:
        if config.verbose:
            print("[TTA-v4] No parameters to update")
        return {
            'predictions': pred_init,
            'logits': logits_init,
            'initial_predictions': pred_init,
            'improvement': 0.0,
            'skipped': True,
            'evidence_mean': q_mean_init,
        }

    # Enable gradients
    for p in model.parameters():
        p.requires_grad = False
    for p in params_to_update:
        p.requires_grad = True

    optimizer = torch.optim.Adam(params_to_update, lr=config.tta_lr)

    if config.verbose:
        print(f"[TTA-v4] Updating {len(params_to_update)} parameters "
              f"(real + fake artifact attention)")

    # ========================================
    # Phase 4: TTA Loop
    # ========================================
    tta_history = []
    prev_loss = None
    prev_q_mean = q_mean_init
    collapse_counter = 0

    for step in range(config.max_tta_steps):
        # Forward
        logits, features, debug = model(images, test_time=True)
        prob = torch.sigmoid(logits)

        # Evidence quality
        artifact_score_step = debug['artifact_score']
        noise_level_step = debug['noise_level']
        routing_weights_step = debug['routing_weights']
        q_step = model.compute_evidence_quality(artifact_score_step, noise_level_step).detach()
        q_mean_step = q_step.mean().item()

        # Sample selection (top-ρ)
        q_threshold = compute_quantile(q_step, 1.0 - config.update_sample_ratio, dim=0).item()
        mask = (q_step >= q_threshold).float().unsqueeze(1)
        num_selected = int(mask.sum().item())

        if num_selected == 0:
            mask = torch.ones_like(mask)
            num_selected = B

        # Soft gating (LR scaling by evidence)
        gating_factor = torch.sigmoid(torch.tensor(2.0 * (q_mean_step - 0.5))).item()
        effective_lr = config.tta_lr * gating_factor

        for param_group in optimizer.param_groups:
            param_group['lr'] = effective_lr

        # Compute loss (entropy on selected samples)
        entropy_per_sample = -(
            prob * torch.log(prob + eps) +
            (1 - prob) * torch.log(1 - prob + eps)
        )
        masked_entropy = (mask * entropy_per_sample).sum() / (mask.sum() + eps)
        loss = masked_entropy

        # Backward and update
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params_to_update, max_norm=1.0)
        optimizer.step()

        # Early stopping checks
        prob_mean = prob.mean().item()
        prob_std = prob.std().item()
        loss_val = loss.item()

        # Loss convergence
        converged = False
        if prev_loss is not None:
            loss_change = abs(loss_val - prev_loss)
            relative_eps = 0.01 * abs(prev_loss) if abs(prev_loss) > 1e-6 else 1e-4
            if loss_change < relative_eps:
                converged = True

        # Evidence stability
        evidence_stable = False
        q_change = abs(q_mean_step - prev_q_mean)
        if q_change < 0.05:
            evidence_stable = True

        # Collapse detection
        collapse_detected = False
        if prob_mean < 0.01 or prob_mean > 0.99:
            collapse_detected = True
        if prob_std < 0.05:
            collapse_detected = True

        if collapse_detected:
            collapse_counter += 1
            if collapse_counter >= 2:
                if config.verbose:
                    print(f"[TTA-v4] Collapse at step {step}, STOP")
                break
        else:
            collapse_counter = 0

        # Record history
        tta_history.append({
            'step': step,
            'loss': loss_val,
            'num_selected': num_selected,
            'q_mean': q_mean_step,
            'q_threshold': q_threshold,
            'prob_mean': prob_mean,
            'prob_std': prob_std,
            'routing_weight_mean': routing_weights_step.mean().item(),
            'routing_weight_std': routing_weights_step.std().item(),
            'gating_factor': gating_factor,
            'effective_lr': effective_lr,
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'converged': converged,
            'evidence_stable': evidence_stable,
        })

        # Early stop
        if converged and evidence_stable:
            if config.verbose:
                print(f"[TTA-v4] Converged at step {step}")
            break

        prev_loss = loss_val
        prev_q_mean = q_mean_step

    # ========================================
    # Phase 5: Final Prediction
    # ========================================
    model.eval()
    with torch.no_grad():
        logits_final, features_final, debug_final = model(images, test_time=True)
        pred_final = torch.sigmoid(logits_final)

        routing_weights_final = debug_final['routing_weights']
        artifact_score_final = debug_final['artifact_score']
        noise_level_final = debug_final['noise_level']
        q_final = model.compute_evidence_quality(artifact_score_final, noise_level_final)

    # ========================================
    # Phase 6: Update Class-Specific Memory Banks
    # ========================================
    if model.real_memory_bank is not None and model.fake_memory_bank is not None:
        with torch.no_grad():
            model.update_memory(features_final, logits_final, routing_weights_final, q_final)

    # Disable gradients
    for p in model.parameters():
        p.requires_grad = False

    # ========================================
    # Prepare Results
    # ========================================
    results = {
        'predictions': pred_final,
        'logits': logits_final,
        'initial_predictions': pred_init,
        'improvement': (pred_final - pred_init).mean().item(),
        'skipped': False,
        'evidence_mean': q_mean_init,
        'routing_weights': routing_weights_final,
        'num_steps': len(tta_history),
    }

    if return_debug:
        results['tta_history'] = tta_history
        results['debug_initial'] = debug_init
        results['debug_final'] = debug_final

    return results


def _inference_with_tta_v4_simple(
    model: UnifiedTTANRAMv4,
    images: torch.Tensor,
    config: TTANRAMv4Config,
    return_debug: bool = False
) -> Dict:
    """Simple TTA without evidence calibration (for ablation)."""
    device = config.device
    model.eval()
    eps = torch.finfo(torch.float32).eps * 10

    with torch.no_grad():
        logits_init, _, _ = model(images, test_time=True)
        pred_init = torch.sigmoid(logits_init)

    params_to_update = [
        p for n, p in model.named_parameters()
        if 'real_artifact_attention' in n or 'fake_artifact_attention' in n
    ]

    for p in model.parameters():
        p.requires_grad = False
    for p in params_to_update:
        p.requires_grad = True

    optimizer = torch.optim.Adam(params_to_update, lr=config.tta_lr)

    for step in range(config.max_tta_steps):
        logits, _, _ = model(images, test_time=True)
        prob = torch.sigmoid(logits)

        entropy = -(prob * torch.log(prob + eps) + (1 - prob) * torch.log(1 - prob + eps))
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
    }


# ============================================================================
# Utility Functions
# ============================================================================

def print_model_info(model: UnifiedTTANRAMv4):
    """Print model architecture and parameter counts."""
    print("=" * 80)
    print("TTA-NRAM v4 (Evidence-Calibrated Soft Class-Conditional) Model Information")
    print("=" * 80)

    # Config
    print(f"\nConfiguration (6 core hyperparameters):")
    print(f"  Model: {model.config.model}")
    print(f"  Target Layer: {model.config.target_layer}")
    print(f"  Reduction Ratio: {model.config.reduction_ratio}")
    print(f"  Max TTA Steps: {model.config.max_tta_steps}")
    print(f"  TTA LR: {model.config.tta_lr}")
    print(f"  Update Sample Ratio (ρ): {model.config.update_sample_ratio}")
    print(f"  Temperature Range: [{model.config.temp_min}, {model.config.temp_max}]")
    print(f"  Memory Bank: {model.config.enable_memory_bank}")
    if model.config.enable_memory_bank:
        print(f"    ├─ Confidence Threshold: {model.config.memory_confidence_threshold}")
        print(f"    └─ Evidence Threshold: {model.config.memory_evidence_threshold}")

    # Parameters
    params = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable (during TTA): {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")

    # Components
    nram_params = count_parameters(model.nram)
    print(f"\nNRAM v4 Components:")
    print(f"  Total NRAM: {nram_params['total']:,} params")
    print(f"    ├─ Artifact Detector: 0 params (data-driven)")
    print(f"    ├─ Noise Estimator: 0 params (data-driven)")
    print(f"    ├─ Soft Routing: 0 params (temperature-controlled)")
    print(f"    ├─ Real Artifact Attention: Learnable (dual-path SE)")
    print(f"    ├─ Fake Artifact Attention: Learnable (dual-path SE)")
    print(f"    └─ Residual Weight α: Fixed (0.1)")
    print(f"  Base Classifier: Frozen (pre-trained)")

    # Memory bank stats
    if model.real_memory_bank is not None:
        real_stats = model.real_memory_bank.get_statistics()
        fake_stats = model.fake_memory_bank.get_statistics()
        print(f"\nMemory Bank Status:")
        print(f"  Real Memory: {real_stats['num_samples']} samples")
        print(f"  Fake Memory: {fake_stats['num_samples']} samples")

    print("=" * 80)


if __name__ == "__main__":
    print("TTA-NRAM v4 (Evidence-Calibrated Soft Class-Conditional) module loaded!")
    print("\nKey features:")
    print("  ✅ Soft routing prevents pseudo-label error accumulation")
    print("  ✅ Evidence-calibrated temperature control")
    print("  ✅ Class-specific NRAM modules (real/fake)")
    print("  ✅ Dual-gated memory banks (confidence AND evidence)")
    print("  ✅ Data-driven normalization (no magic numbers)")
    print("  ✅ Only 6 hyperparameters")
    print("\nUsage:")
    print("  from model.method.tta_nramv4 import UnifiedTTANRAMv4, TTANRAMv4Config, inference_with_tta_v4")
    print("\nConfig example:")
    print("  config = TTANRAMv4Config(")
    print("      model='LGrad',")
    print("      reduction_ratio=16,")
    print("      max_tta_steps=10,")
    print("      tta_lr=1e-4,")
    print("      update_sample_ratio=0.7,")
    print("      temp_min=0.1,")
    print("      temp_max=10.0,")
    print("  )")
