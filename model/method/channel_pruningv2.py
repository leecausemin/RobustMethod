"""
Channel Pruning v2 (CPv2) - Percentile-Based Robust Channel Gating

핵심 개선사항 (vs CPv1):
✅ Batch-level variance 문제 해결
✅ Percentile-based gating (상대적 중요도)
✅ Z-score normalization (robust to variance)
✅ No temperature/bias (단순하고 안정적)
✅ Learnable pruning ratio

방법론:
1. Pre-compute: Real/Fake separated statistics (mean, std)
2. Test time:
   - Z-score deviation 계산: (curr - ref) / ref_std
   - Robustness score: min(|z_real|, |z_fake|) - 작을수록 robust
   - Channel importance: artifact_disc / robustness
3. Percentile-based gating:
   - Importance의 상위 k% 채널만 keep
   - k는 learnable parameter

주요 차이점:
CPv1: score = disc / sensitivity → sigmoid → gate (0~1)
CPv2: importance = disc / z_score → percentile thresholding → gate (0 or 1)

Based on research:
- Channel Gating Neural Networks (NeurIPS 2019)
- Channel-Selective Normalization for Test-Time Adaptation (2024)
- Sensitivity-Guided Pruning (2024)
"""

import copy
from dataclasses import dataclass
from typing import Union, Optional, Literal, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningStats:
    """
    Online statistics computation using Welford's algorithm
    Computes running mean and variance without storing all data
    """
    def __init__(self, num_channels: int, device: str = "cuda"):
        self.device = device
        self.count = 0
        self.mean = torch.zeros(num_channels, device=device)
        self.M2 = torch.zeros(num_channels, device=device)  # Sum of squared differences

    def update(self, batch: torch.Tensor):
        """
        Update running statistics with new batch (VECTORIZED - FAST!)

        Args:
            batch: [B, C, H, W] or [B, C] tensor
        """
        # Reduce spatial dimensions if needed
        if batch.dim() == 4:
            # [B, C, H, W] -> [N, C] where N = B*H*W
            B, C, H, W = batch.shape
            batch_flat = batch.permute(0, 2, 3, 1).reshape(-1, C)
        elif batch.dim() == 2:
            # [B, C] -> already flat
            batch_flat = batch
        else:
            raise ValueError(f"Unexpected batch shape: {batch.shape}")

        # Batch statistics (GPU vectorized!)
        n_new = batch_flat.shape[0]
        mean_new = batch_flat.mean(dim=0)
        var_new = batch_flat.var(dim=0, unbiased=False)  # Population variance

        # Welford's algorithm for combining two sets of statistics
        n_old = self.count
        n_total = n_old + n_new

        if n_old == 0:
            # First batch
            self.mean = mean_new.clone()
            self.M2 = var_new * n_new
            self.count = n_new
        else:
            # Combine statistics (parallel algorithm)
            delta = mean_new - self.mean
            self.mean = (n_old * self.mean + n_new * mean_new) / n_total
            self.M2 = self.M2 + var_new * n_new + delta**2 * n_old * n_new / n_total
            self.count = n_total

    def get_stats(self) -> Dict[str, torch.Tensor]:
        """Get mean, variance, and std"""
        if self.count < 2:
            var = torch.zeros_like(self.mean)
        else:
            var = self.M2 / self.count

        std = var.sqrt()

        return {
            'mean': self.mean.clone(),
            'var': var.clone(),
            'std': std.clone(),
        }


@dataclass
class CPv2Config:
    """Configuration for Channel Pruning v2 (Percentile-Based Robust Gating)"""

    # Model type
    model: Literal["LGrad", "NPR"] = "LGrad"

    # Target layers to apply channel gating
    # None = automatically select all layers with statistics
    target_layers: Optional[list[str]] = None

    # Percentile-based gating
    keep_ratio: float = 0.7  # Keep top 70% channels (learnable)
    use_learnable_keep_ratio: bool = False  # Make keep_ratio learnable

    # Gating type
    gating_type: Literal["hard", "soft"] = "hard"  # Hard gating recommended
    soft_temperature: float = 10.0  # Only for soft gating

    # Z-score normalization
    use_zscore: bool = True  # Recommended
    zscore_eps: float = 1e-3  # Epsilon for std

    # Robustness metric
    robustness_method: Literal["min", "avg", "max"] = "min"
    # "min": min(|z_real|, |z_fake|) - RECOMMENDED
    # "avg": average
    # "max": max (conservative)

    # Importance metric
    importance_method: Literal["div", "sub"] = "div"
    # "div": artifact_disc / robustness
    # "sub": artifact_disc - robustness

    device: str = "cuda"

    def __post_init__(self):
        """Validate configuration"""
        if not (0 < self.keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be in (0, 1]")
        if self.zscore_eps <= 0:
            raise ValueError("zscore_eps must be positive")


def compute_separated_statistics_v2(
    model,
    dataloader,
    target_layers: Optional[list[str]] = None,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
    """
    Compute separated channel-wise statistics for Real and Fake (MEMORY EFFICIENT)

    Same as v1 but ensures std is computed
    """
    model.eval()
    model.to(device)

    # Determine target layers
    if target_layers is None:
        target_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                # Skip grad_model layers for LGrad
                if 'grad_model' not in name:
                    target_layers.append(name)

    print(f"[CPv2] Computing separated statistics for {len(target_layers)} layers (ONLINE MODE)...")

    # Storage for current batch activations only
    current_activations = {}

    # Initialize running statistics for each layer
    running_stats = {}
    for layer_name in target_layers:
        running_stats[layer_name] = {
            'real': None,  # Will be initialized on first batch
            'fake': None,
        }

    # Register hooks
    handles = []

    def make_hook(layer_name):
        def hook(module, input, output):
            # Store current batch activation only (will be deleted after processing)
            if output.dim() >= 2:
                current_activations[layer_name] = output.detach()
        return hook

    for name in target_layers:
        module = dict(model.named_modules())[name]
        handle = module.register_forward_hook(make_hook(name))
        handles.append(handle)

    # Process batches
    total_batches = max_batches if max_batches is not None else len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=total_batches, desc="Computing separated statistics v2 (online)")

    for batch_idx, batch in pbar:
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Handle different batch formats
        if isinstance(batch, (tuple, list)):
            images = batch[0]
            labels = batch[1]  # MUST have labels!
        else:
            raise ValueError("DataLoader must provide (images, labels) tuples for separated statistics!")

        images = images.to(device)
        labels = labels.to(device)

        # Clear current activations
        current_activations.clear()

        # Forward pass (hooks will capture activations)
        try:
            with torch.no_grad():
                _ = model(images)
        except:
            try:
                _ = model(images)
            except:
                if hasattr(model, 'model'):
                    _ = model.model(images)
                else:
                    raise

        # Process activations for each layer
        for layer_name in target_layers:
            if layer_name not in current_activations:
                continue

            acts = current_activations[layer_name]  # [B, C, H, W] or [B, C]

            # Initialize running stats on first batch
            if running_stats[layer_name]['real'] is None:
                C = acts.shape[1]
                running_stats[layer_name]['real'] = RunningStats(C, device)
                running_stats[layer_name]['fake'] = RunningStats(C, device)

            # Split by label
            real_mask = (labels == 0)
            fake_mask = (labels == 1)

            # Update running statistics
            if real_mask.any():
                real_acts = acts[real_mask]
                running_stats[layer_name]['real'].update(real_acts)

            if fake_mask.any():
                fake_acts = acts[fake_mask]
                running_stats[layer_name]['fake'].update(fake_acts)

        # Clear activations and free memory
        current_activations.clear()
        del images, labels
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Collect final statistics
    separated_stats = {}

    for layer_name in target_layers:
        if running_stats[layer_name]['real'] is None:
            print(f"  Warning: No activations collected for {layer_name}, skipping...")
            continue

        real_stats = running_stats[layer_name]['real'].get_stats()
        fake_stats = running_stats[layer_name]['fake'].get_stats()

        if running_stats[layer_name]['real'].count == 0 or running_stats[layer_name]['fake'].count == 0:
            print(f"  Warning: {layer_name} has no real or fake samples, skipping...")
            continue

        # Move to CPU for storage (final stats are small!)
        separated_stats[layer_name] = {
            'real': {k: v.cpu() for k, v in real_stats.items()},
            'fake': {k: v.cpu() for k, v in fake_stats.items()},
        }

        # Print info
        C = len(real_stats['mean'])
        real_mean_range = real_stats['mean'].min().item(), real_stats['mean'].max().item()
        fake_mean_range = fake_stats['mean'].min().item(), fake_stats['mean'].max().item()

        print(f"  {layer_name}: C={C}")
        print(f"    Real: mean=[{real_mean_range[0]:.4f}, {real_mean_range[1]:.4f}], std_mean={real_stats['std'].mean():.4f}")
        print(f"    Fake: mean=[{fake_mean_range[0]:.4f}, {fake_mean_range[1]:.4f}], std_mean={fake_stats['std'].mean():.4f}")

        # Artifact signature
        artifact_sig = (fake_stats['mean'] - real_stats['mean']).abs()
        print(f"    Artifact signature: mean={artifact_sig.mean():.4f}, max={artifact_sig.max():.4f}")

    print(f"[CPv2] Separated statistics computed for {len(separated_stats)} layers")

    return separated_stats


class RobustChannelGating(nn.Module):
    """
    Percentile-Based Robust Channel Gating Module (CPv2)

    핵심 개선:
    - Z-score normalization으로 batch variance 처리
    - Percentile-based thresholding으로 상대적 중요도
    - No temperature/bias (단순하고 robust)

    방법:
    1. Z-score deviation: (curr - ref) / ref_std
    2. Robustness: min(|z_real|, |z_fake|) - 작을수록 robust to corruption
    3. Importance: artifact_disc / robustness - 클수록 중요
    4. Gating: top-k% channels만 keep
    """

    def __init__(
        self,
        num_channels: int,
        layer_name: str,
        separated_stats: Dict[str, Dict[str, torch.Tensor]],
        config: CPv2Config,
    ):
        super().__init__()
        self.layer_name = layer_name
        self.separated_stats = separated_stats
        self.cfg = config

        # Pre-compute artifact discriminability
        artifact_signature = torch.abs(
            separated_stats['fake']['mean'] - separated_stats['real']['mean']
        )
        self.register_buffer('artifact_discriminability', artifact_signature)

        # Learnable keep ratio (optional)
        if config.use_learnable_keep_ratio:
            # Logit space: sigmoid(logit) = keep_ratio
            initial_logit = torch.log(torch.tensor(config.keep_ratio) / (1 - config.keep_ratio + 1e-8))
            self.keep_ratio_logit = nn.Parameter(initial_logit)
        else:
            self.register_buffer('keep_ratio', torch.tensor(config.keep_ratio))

    def compute_robustness(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute robustness score using Z-score deviation

        핵심: Batch-level variance를 고려한 정규화
        - Z-score = (curr - ref_mean) / ref_std
        - Robustness = min(|z_real|, |z_fake|)
        - 작을수록 robust (corruption에 덜 민감)

        Args:
            x: [B, C, H, W] or [B, C]

        Returns:
            robustness: [B, C] - 작을수록 robust
        """
        # Current batch statistics
        if x.dim() == 4:
            curr_mean = x.mean(dim=[2, 3])  # [B, C]
        elif x.dim() == 2:
            curr_mean = x  # [B, C]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # Reference statistics (from clean data)
        real_mean = self.separated_stats['real']['mean'].to(x.device)  # [C]
        fake_mean = self.separated_stats['fake']['mean'].to(x.device)  # [C]
        real_std = self.separated_stats['real']['std'].to(x.device)  # [C]
        fake_std = self.separated_stats['fake']['std'].to(x.device)  # [C]

        if self.cfg.use_zscore:
            # Z-score normalization (batch variance 고려!)
            z_real = torch.abs((curr_mean - real_mean) / (real_std + self.cfg.zscore_eps))  # [B, C]
            z_fake = torch.abs((curr_mean - fake_mean) / (fake_std + self.cfg.zscore_eps))  # [B, C]
        else:
            # Simple deviation (for ablation)
            z_real = torch.abs(curr_mean - real_mean)
            z_fake = torch.abs(curr_mean - fake_mean)

        # Combine based on robustness method
        if self.cfg.robustness_method == "min":
            # RECOMMENDED: Minimum deviation
            # Clean image → close to at least one reference → small
            # Corrupted image → far from both → large
            robustness = torch.min(z_real, z_fake)  # [B, C]
        elif self.cfg.robustness_method == "avg":
            robustness = (z_real + z_fake) / 2
        elif self.cfg.robustness_method == "max":
            robustness = torch.max(z_real, z_fake)
        else:
            raise ValueError(f"Unknown robustness_method: {self.cfg.robustness_method}")

        return robustness  # [B, C]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply percentile-based robust channel gating

        Args:
            x: [B, C, H, W] or [B, C]

        Returns:
            x_gated: [B, C, H, W] or [B, C] - gated output
            gate: [C] - binary gate values (0 or 1 for hard, 0~1 for soft)
        """
        # Step 1: Robustness score (작을수록 robust)
        robustness = self.compute_robustness(x)  # [B, C]

        # Step 2: Artifact discriminability (클수록 중요)
        disc = self.artifact_discriminability.unsqueeze(0)  # [1, C]

        # Step 3: Channel importance
        epsilon = 1e-6
        if self.cfg.importance_method == "div":
            # RECOMMENDED: importance = disc / robustness
            # High importance = high discriminability + low corruption sensitivity
            importance = disc / (robustness + epsilon)  # [B, C]
        elif self.cfg.importance_method == "sub":
            # Alternative: importance = disc - robustness
            importance = disc - robustness
        else:
            raise ValueError(f"Unknown importance_method: {self.cfg.importance_method}")

        # Step 4: Aggregate across batch (channel-level gating)
        importance_agg = importance.mean(dim=0)  # [C]

        # Step 5: Percentile-based gating
        if self.cfg.use_learnable_keep_ratio:
            keep_ratio = torch.sigmoid(self.keep_ratio_logit)
        else:
            keep_ratio = self.keep_ratio

        # Compute threshold (k-th percentile)
        k = int((1 - keep_ratio) * importance_agg.numel())
        k = max(0, min(k, importance_agg.numel() - 1))  # Clamp

        if k == 0:
            # Keep all channels
            threshold = importance_agg.min() - 1
        else:
            threshold = torch.kthvalue(importance_agg, k + 1)[0]

        # Apply gating
        if self.cfg.gating_type == "hard":
            # Binary gate: 1 if importance >= threshold, else 0
            gate = (importance_agg >= threshold).float()  # [C]
        else:  # soft
            # Soft gate with temperature
            gate_logits = (importance_agg - threshold) * self.cfg.soft_temperature
            gate = torch.sigmoid(gate_logits)  # [C]

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


class UnifiedChannelPruningV2(nn.Module):
    """
    Unified Channel Pruning v2 for both LGrad and NPR

    핵심 개선 (vs v1):
    - Percentile-based robust gating
    - Z-score normalization
    - No temperature/bias tuning needed
    - Simpler and more robust

    사용법:
        # 1. Separated statistics 수집
        stats = compute_separated_statistics_v2(model, clean_loader)

        # 2. Config 설정
        config = CPv2Config(
            model="LGrad",
            keep_ratio=0.7,  # Keep top 70%
            use_zscore=True,
            gating_type="hard",
        )

        # 3. Model 생성
        model = UnifiedChannelPruningV2(base_model, stats, config)

        # 4. Inference
        probs = model.predict_proba(images)
    """

    def __init__(
        self,
        base_model,
        separated_stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
        config: CPv2Config,
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

        # Store statistics
        self.separated_stats = separated_stats

        # Setup gating modules
        self.gates = nn.ModuleDict()
        self.gate_name_mapping = {}
        self._setup_gates()

        print(f"[CPv2] Initialized for {config.model}")
        print(f"[CPv2] Target layers: {len(self.gates)}")
        print(f"[CPv2] Keep ratio: {config.keep_ratio:.2%} (learnable={config.use_learnable_keep_ratio})")
        print(f"[CPv2] Gating type: {config.gating_type}")
        print(f"[CPv2] Z-score normalization: {config.use_zscore}")

    def _setup_gates(self):
        """Setup gating modules for target layers"""

        # Determine target layers
        target_layers = self.cfg.target_layers
        if target_layers is None:
            target_layers = list(self.separated_stats.keys())

        # Validate
        for layer_name in target_layers:
            if layer_name not in self.separated_stats:
                raise ValueError(f"Layer {layer_name} not found in separated_stats")

        # Create gating modules
        for layer_name in target_layers:
            # Get layer module
            try:
                layer_module = dict(self.model.named_modules())[layer_name]
            except KeyError:
                print(f"  Warning: Layer {layer_name} not found in model, skipping...")
                continue

            # Get number of channels from statistics
            num_channels = len(self.separated_stats[layer_name]['real']['mean'])

            # Create gating module
            gate_module = RobustChannelGating(
                num_channels=num_channels,
                layer_name=layer_name,
                separated_stats=self.separated_stats[layer_name],
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

            # Print info
            artifact_disc = gate_module.artifact_discriminability
            print(f"  {layer_name} (C={num_channels}):")
            print(f"    Artifact disc: mean={artifact_disc.mean():.4f}, max={artifact_disc.max():.4f}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with robust channel gating"""
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
