"""
Channel Pruning v3 (CPv3) - Quality-Based Selective Channel Gating

핵심 개선사항 (vs v1/v2):
✅ Intrinsic channel quality 기반 (노이즈 타입 무관!)
✅ Quality = Discriminability / sqrt(Variance)
✅ Two-stage hybrid gating (extreme outlier만 prune, 나머지는 soft)
✅ Clean data만 필요 (일반화 가능)
✅ Simple & Elegant

사용자 관찰:
"몇 개의 채널이 유독 노이즈에 민감함"
→ 소수(5~10%) 채널만 문제
→ 나머지(90~95%)는 robust

방법론:
1. Pre-compute (clean data만):
   - Artifact discriminability: |fake_mean - real_mean|
   - Intrinsic variance: var(clean_features)
   - Quality: disc / sqrt(var)

2. Two-stage gating:
   - Stage 1: Bottom 5% quality → Hard prune (gate=0)
   - Stage 2: 5~20% quality → Soft weight (gate=0.5~1.0)
   - Stage 3: Top 80% quality → Keep (gate=1.0)

3. Test time:
   - Pre-computed quality 사용
   - 노이즈 타입 무관하게 작동!

장점:
- 새로운 노이즈에도 일반화
- Clean data만 필요
- Artifact detection 채널 보존
- 직관적이고 simple
"""

import copy
from dataclasses import dataclass
from typing import Union, Optional, Literal, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CPv3Config:
    """Configuration for Channel Pruning v3 (Quality-Based Selective Gating)"""

    # Model type
    model: Literal["LGrad", "NPR"] = "LGrad"

    # Target layers to apply channel gating
    target_layers: Optional[list[str]] = None

    # Quality-based thresholds (percentiles)
    low_quality_percentile: float = 0.05   # Bottom 5% → hard prune
    medium_quality_percentile: float = 0.20  # Bottom 20% → soft weight

    # Gating parameters
    medium_quality_min_weight: float = 0.3  # Minimum weight for medium quality channels

    # Optional: Use artifact discriminability threshold
    use_discriminability_threshold: bool = True
    discriminability_min: float = 0.001  # Channels with disc < this are always pruned

    device: str = "cuda"

    def __post_init__(self):
        """Validate configuration"""
        if not (0 < self.low_quality_percentile < self.medium_quality_percentile < 1.0):
            raise ValueError("Must have: 0 < low < medium < 1.0")


def compute_channel_quality(
    separated_stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute intrinsic channel quality from separated statistics

    Quality = Artifact Discriminability / sqrt(Intrinsic Variance)

    High quality: High discriminability + Low variance → Robust & useful
    Low quality: Low discriminability + High variance → Unstable & useless

    Args:
        separated_stats: Pre-computed separated statistics (from v1 or v2)

    Returns:
        quality_stats: {
            layer_name: {
                'quality': [C],
                'discriminability': [C],
                'variance': [C],
            }
        }

    Example:
        >>> stats = torch.load("separated_stats_lgrad_progan.pth")
        >>> quality = compute_channel_quality(stats)
        >>> print(quality['classifier.layer4.2.bn3']['quality'])
    """
    print(f"[CPv3] Computing channel quality for {len(separated_stats)} layers...")

    quality_stats = {}

    for layer_name, stats in separated_stats.items():
        # 1. Artifact discriminability (이미 계산됨!)
        real_mean = stats['real']['mean']
        fake_mean = stats['fake']['mean']
        discriminability = (fake_mean - real_mean).abs()  # [C]

        # 2. Intrinsic variance (clean data 전체의 variance)
        # Real과 Fake 합쳐서 전체 variance 추정
        real_var = stats['real']['var']
        fake_var = stats['fake']['var']
        intrinsic_var = (real_var + fake_var) / 2  # [C]

        # 3. Channel quality score
        # High quality = High disc + Low var
        # Low quality = Low disc + High var
        quality = discriminability / (intrinsic_var.sqrt() + 1e-6)  # [C]

        quality_stats[layer_name] = {
            'quality': quality,
            'discriminability': discriminability,
            'variance': intrinsic_var,
        }

        # Print info
        C = len(quality)
        quality_sorted = torch.sort(quality)[0]
        q_min = quality_sorted[0].item()
        q_25 = quality_sorted[int(C * 0.25)].item()
        q_50 = quality_sorted[int(C * 0.50)].item()
        q_75 = quality_sorted[int(C * 0.75)].item()
        q_max = quality_sorted[-1].item()

        print(f"  {layer_name} (C={C}):")
        print(f"    Quality: min={q_min:.4f}, Q25={q_25:.4f}, median={q_50:.4f}, Q75={q_75:.4f}, max={q_max:.4f}")
        print(f"    Disc: mean={discriminability.mean():.4f}, Var: mean={intrinsic_var.mean():.4f}")

    print(f"[CPv3] Channel quality computed for {len(quality_stats)} layers")

    return quality_stats


class QualityBasedHybridGating(nn.Module):
    """
    Quality-Based Two-Stage Hybrid Gating Module

    핵심 아이디어:
    - 대부분(80%) 채널: High quality → Keep (gate=1.0)
    - 소수(15%) 채널: Medium quality → Soft weight (gate=0.3~1.0)
    - 극소수(5%) 채널: Low quality → Hard prune (gate=0)

    방법:
    1. Quality score (pre-computed)
    2. Percentile-based thresholding
    3. Two-stage gating:
       - Stage 1: Bottom 5% → gate=0
       - Stage 2: 5~20% → gate=0.3~1.0 (linear)
       - Stage 3: Top 80% → gate=1.0
    """

    def __init__(
        self,
        num_channels: int,
        layer_name: str,
        quality_stats: Dict[str, torch.Tensor],
        config: CPv3Config,
    ):
        super().__init__()
        self.layer_name = layer_name
        self.cfg = config

        # Quality and related stats
        quality = quality_stats['quality']  # [C]
        discriminability = quality_stats['discriminability']  # [C]
        variance = quality_stats['variance']  # [C]

        self.register_buffer('quality', quality)
        self.register_buffer('discriminability', discriminability)
        self.register_buffer('variance', variance)

        # Compute thresholds based on percentiles
        quality_sorted = torch.sort(quality)[0]
        low_idx = int(num_channels * config.low_quality_percentile)
        medium_idx = int(num_channels * config.medium_quality_percentile)

        self.register_buffer('low_quality_threshold', quality_sorted[low_idx])
        self.register_buffer('medium_quality_threshold', quality_sorted[medium_idx])

        # Optional: Discriminability threshold
        if config.use_discriminability_threshold:
            self.register_buffer('disc_min', torch.tensor(config.discriminability_min))
        else:
            self.register_buffer('disc_min', torch.tensor(0.0))

        # Pre-compute static gate
        self._compute_static_gate()

    def _compute_static_gate(self):
        """Pre-compute static gate based on quality"""
        quality = self.quality

        # Stage 1: Low quality (bottom 5%) → Hard prune
        # Also prune channels with very low discriminability
        low_quality_mask = (quality < self.low_quality_threshold) | \
                          (self.discriminability < self.disc_min)

        # Stage 2: Medium quality (5~20%) → Soft weight
        medium_quality_mask = (quality >= self.low_quality_threshold) & \
                             (quality < self.medium_quality_threshold) & \
                             (self.discriminability >= self.disc_min)

        # Stage 3: High quality (top 80%) → Keep
        high_quality_mask = (quality >= self.medium_quality_threshold) & \
                           (self.discriminability >= self.disc_min)

        # Initialize gate
        gate = torch.zeros_like(quality)

        # Low quality: gate = 0
        gate[low_quality_mask] = 0.0

        # Medium quality: gate = linear interpolation (min_weight ~ 1.0)
        if medium_quality_mask.any():
            # Normalize quality to [0, 1] within medium range
            quality_normalized = (quality - self.low_quality_threshold) / \
                               (self.medium_quality_threshold - self.low_quality_threshold + 1e-6)
            quality_normalized = torch.clamp(quality_normalized, 0, 1)

            # Linear interpolation: min_weight ~ 1.0
            gate[medium_quality_mask] = self.cfg.medium_quality_min_weight + \
                                       (1.0 - self.cfg.medium_quality_min_weight) * quality_normalized[medium_quality_mask]

        # High quality: gate = 1.0
        gate[high_quality_mask] = 1.0

        self.register_buffer('static_gate', gate)

        # Print statistics
        num_low = low_quality_mask.sum().item()
        num_medium = medium_quality_mask.sum().item()
        num_high = high_quality_mask.sum().item()
        num_total = len(quality)

        print(f"    Gate distribution:")
        print(f"      Low quality (pruned): {num_low}/{num_total} ({num_low/num_total*100:.1f}%)")
        print(f"      Medium quality (soft): {num_medium}/{num_total} ({num_medium/num_total*100:.1f}%)")
        print(f"      High quality (keep): {num_high}/{num_total} ({num_high/num_total*100:.1f}%)")
        print(f"      Gate mean: {gate.mean():.4f}, min: {gate.min():.4f}, max: {gate.max():.4f}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply quality-based gating

        Args:
            x: [B, C, H, W] or [B, C]

        Returns:
            x_gated: Gated output
            gate: Gate values [C]
        """
        # Use pre-computed static gate
        gate = self.static_gate  # [C]

        # Broadcast to input shape
        if x.dim() == 4:
            gate_broadcast = gate.view(1, -1, 1, 1)  # [1, C, 1, 1]
        elif x.dim() == 2:
            gate_broadcast = gate.unsqueeze(0)  # [1, C]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # Apply gating
        x_gated = x * gate_broadcast

        return x_gated, gate


class UnifiedChannelPruningV3(nn.Module):
    """
    Unified Channel Pruning v3 for both LGrad and NPR

    핵심 (vs v1/v2):
    - Quality-based selective gating
    - 대부분 채널 유지, 소수만 제거
    - 노이즈 타입 무관하게 일반화
    - Clean data만 필요

    사용법:
        # 1. Separated statistics 계산 (v1/v2와 동일)
        stats = torch.load("separated_stats_lgrad_progan.pth")

        # 2. Channel quality 계산
        quality = compute_channel_quality(stats)

        # 3. Config
        config = CPv3Config(
            model="LGrad",
            low_quality_percentile=0.05,   # Bottom 5% prune
            medium_quality_percentile=0.20,  # Bottom 20% soft weight
        )

        # 4. Model 생성
        model = UnifiedChannelPruningV3(base_model, quality, config)

        # 5. Inference
        probs = model.predict_proba(images)
    """

    def __init__(
        self,
        base_model,
        quality_stats: Dict[str, Dict[str, torch.Tensor]],
        config: CPv3Config,
    ):
        super().__init__()
        self.cfg = config
        self.device = config.device

        # Deep copy base model
        self.model = copy.deepcopy(base_model)
        self.model.to(self.device)

        # Validate
        if config.model not in ["LGrad", "NPR"]:
            raise ValueError(f"Unsupported model type: {config.model}")

        # Store quality stats
        self.quality_stats = quality_stats

        # Setup gating modules
        self.gates = nn.ModuleDict()
        self.gate_name_mapping = {}
        self._setup_gates()

        print(f"[CPv3] Initialized for {config.model}")
        print(f"[CPv3] Target layers: {len(self.gates)}")
        print(f"[CPv3] Low quality threshold: {config.low_quality_percentile:.1%}")
        print(f"[CPv3] Medium quality threshold: {config.medium_quality_percentile:.1%}")

    def _setup_gates(self):
        """Setup gating modules for target layers"""

        # Determine target layers
        target_layers = self.cfg.target_layers
        if target_layers is None:
            target_layers = list(self.quality_stats.keys())

        # Validate
        for layer_name in target_layers:
            if layer_name not in self.quality_stats:
                raise ValueError(f"Layer {layer_name} not found in quality_stats")

        # Create gating modules
        for layer_name in target_layers:
            # Get layer module
            try:
                layer_module = dict(self.model.named_modules())[layer_name]
            except KeyError:
                print(f"  Warning: Layer {layer_name} not found in model, skipping...")
                continue

            # Get number of channels
            num_channels = len(self.quality_stats[layer_name]['quality'])

            # Create gating module
            gate_module = QualityBasedHybridGating(
                num_channels=num_channels,
                layer_name=layer_name,
                quality_stats=self.quality_stats[layer_name],
                config=self.cfg,
            )

            gate_module = gate_module.to(self.device)

            # Sanitize name
            sanitized_name = layer_name.replace('.', '_')
            self.gates[sanitized_name] = gate_module
            self.gate_name_mapping[sanitized_name] = layer_name

            # Install hook
            def make_hook(gate):
                def hook(module, input, output):
                    gated_output, gate_weights = gate(output)
                    return gated_output
                return hook

            layer_module.register_forward_hook(make_hook(gate_module))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quality-based gating"""
        self.model.eval()

        if self.cfg.model == "LGrad":
            logits = self.model(x)
        else:  # NPR
            logits = self.model(x)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict real/fake labels"""
        self.model.eval()
        logits = self.forward(x)
        return (torch.sigmoid(logits) > 0.5).long().squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probability of being fake"""
        self.model.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(1)

    def get_gate_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get gating statistics for analysis"""
        stats = {}

        for sanitized_name, gate in self.gates.items():
            original_name = self.gate_name_mapping[sanitized_name]
            gate_values = gate.static_gate

            stats[original_name] = {
                'mean': gate_values.mean().item(),
                'min': gate_values.min().item(),
                'max': gate_values.max().item(),
                'num_pruned': (gate_values == 0).sum().item(),
                'num_soft': ((gate_values > 0) & (gate_values < 1.0)).sum().item(),
                'num_keep': (gate_values == 1.0).sum().item(),
            }

        return stats
