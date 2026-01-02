"""
Channel Pruning v1 (CPv1) - Gradient-Pattern-Aware Channel Pruning for LGrad

핵심 아이디어 (IMPROVED):
LGrad의 gradient pattern detection을 보존하면서 noise-robust한 channel만 유지한다.
- Real clean과 Fake clean statistics를 분리해서 저장 (gradient pattern 차이)
- Test time에 LABEL-FREE minimum deviation으로 gradient corruption sensitivity 측정
- Artifact discriminability / Gradient corruption sensitivity 기반 channel scoring
- High score channel 유지 (robust gradient pattern detection)
- Low score channel pruning (noise-sensitive, poor discrimination)

방법론 (IMPROVED):
1. Pre-compute: Real clean과 Fake clean의 분리된 channel statistics
   - Real gradient features vs Fake gradient features
2. Test time: Gradient corruption sensitivity 계산 (LABEL-FREE!)
   - min(|curr - real_stats|, |curr - fake_stats|)
   - Clean: close to at least one reference → small
   - Noisy: far from both references → large
3. Artifact discriminability (pre-computed) 사용
   - |fake_stats - real_stats| = gradient pattern difference
4. Channel score = discriminability / sensitivity
   - High: Strong gradient pattern detection + Low corruption sensitivity
5. Score 기반 channel gating/pruning (with batch aggregation)
   - Stable channel-level gating

주요 개선:
✅ Pseudo-label 제거 (label-free minimum deviation)
✅ Gradient corruption에 robust한 채널 선택
✅ Batch aggregation으로 안정적인 gating
✅ LGrad의 two-stage architecture에 최적화
"""

import copy
from dataclasses import dataclass
from typing import Union, Optional, Literal, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CPv1Config:
    """Configuration for Channel Pruning v1 (Gradient-Pattern-Aware)"""

    # Model type
    model: Literal["LGrad", "NPR"] = "LGrad"

    # Target layers to apply channel gating
    # None = automatically select all layers with statistics
    target_layers: Optional[list[str]] = None

    # Channel gating parameters
    temperature_init: float = 1.0  # Initial temperature (learnable)
    use_learnable_temperature: bool = True
    use_channel_bias: bool = True  # Channel-specific bias

    # Sensitivity computation (NEW: Label-free minimum deviation)
    sensitivity_method: Literal["min", "avg", "max"] = "min"
    # "min": Minimum deviation (RECOMMENDED for LGrad)
    # "avg": Average deviation
    # "max": Maximum deviation (conservative)

    deviation_metric: Literal["mean+var", "mean", "var"] = "mean"
    normalize_deviation: bool = False

    # Batch aggregation (NEW: for stable channel-level gating)
    enable_batch_aggregation: bool = True  # Recommended for LGrad
    aggregation_method: Literal["mean", "median", "min"] = "mean"

    # Gating type
    gating_type: Literal["soft", "hard"] = "soft"
    hard_threshold: float = 0.5  # Hard gating threshold (gating_type="hard"일 때)

    # Pruning ratio control (optional)
    enable_pruning_ratio_control: bool = False
    target_pruning_ratio: float = 0.3  # Target ratio of pruned channels (0~1)

    # Test-time adaptation
    enable_adaptation: bool = False
    adaptation_lr: float = 1e-4
    adaptation_loss: Literal["entropy", "consistency"] = "entropy"

    device: str = "cuda"

    def __post_init__(self):
        """Validate configuration"""
        if self.temperature_init <= 0:
            raise ValueError("temperature_init must be positive")
        if self.gating_type == "hard" and not (0 < self.hard_threshold < 1):
            raise ValueError("hard_threshold must be in (0, 1)")


def compute_separated_statistics(
    model,
    dataloader,
    target_layers: Optional[list[str]] = None,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
    """
    Compute separated channel-wise statistics for Real and Fake

    Args:
        model: Pre-trained model (LGrad or NPR)
        dataloader: DataLoader with clean images (MUST have labels!)
        target_layers: List of layer names to collect statistics
        device: Device to use
        max_batches: Maximum number of batches to process

    Returns:
        separated_stats: {
            layer_name: {
                'real': {'mean': [C], 'var': [C], 'std': [C]},
                'fake': {'mean': [C], 'var': [C], 'std': [C]},
            }
        }

    Example:
        >>> from torch.utils.data import DataLoader
        >>> clean_loader = DataLoader(clean_dataset, batch_size=32)  # MUST have labels!
        >>> stats = compute_separated_statistics(lgrad_model, clean_loader)
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

    print(f"[CPv1] Computing separated statistics for {len(target_layers)} layers...")

    # Storage for activations (will separate by label after collection)
    all_activations = {name: [] for name in target_layers}
    all_labels = []

    # Register hooks
    handles = []

    def make_hook(layer_name):
        def hook(module, input, output):
            # output: [B, C, H, W] or [B, C]
            # Store on CPU immediately to save GPU memory
            if output.dim() >= 2:
                all_activations[layer_name].append(output.detach().cpu())
        return hook

    for name in target_layers:
        module = dict(model.named_modules())[name]
        handle = module.register_forward_hook(make_hook(name))
        handles.append(handle)

    # Collect activations
    total_batches = max_batches if max_batches is not None else len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=total_batches, desc="Computing separated statistics")

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
        labels = labels.cpu()  # Keep on CPU

        # Forward pass
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

        # Store labels for this batch
        all_labels.append(labels)

        # Clear GPU memory
        del images
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Concatenate all labels
    all_labels_tensor = torch.cat(all_labels, dim=0)  # [Total_N]

    # Separate activations by label
    separated_stats = {}

    for layer_name in target_layers:
        if len(all_activations[layer_name]) == 0:
            print(f"  Warning: No activations collected for {layer_name}, skipping...")
            continue

        # Concatenate all activations
        all_acts = torch.cat(all_activations[layer_name], dim=0)  # [Total_N, C, ...]

        # Check shape match
        if all_acts.shape[0] != all_labels_tensor.shape[0]:
            print(f"  Warning: Shape mismatch for {layer_name}: acts={all_acts.shape[0]}, labels={all_labels_tensor.shape[0]}, skipping...")
            continue

        # Split by label
        real_mask = (all_labels_tensor == 0)
        fake_mask = (all_labels_tensor == 1)

        all_real = all_acts[real_mask]  # [N_real, C, ...]
        all_fake = all_acts[fake_mask]  # [N_fake, C, ...]

        if len(all_real) == 0 or len(all_fake) == 0:
            print(f"  Warning: {layer_name} has no real or fake samples, skipping...")
            continue

        # Compute statistics per label
        stats = {'real': {}, 'fake': {}}

        for label_name, acts in [('real', all_real), ('fake', all_fake)]:
            if acts.dim() == 4:
                # Spatial: [N, C, H, W]
                mean = acts.mean(dim=[0, 2, 3])
                var = acts.var(dim=[0, 2, 3])
            elif acts.dim() == 2:
                # Fully connected: [N, C]
                mean = acts.mean(dim=0)
                var = acts.var(dim=0)
            else:
                print(f"  Warning: Unexpected shape {acts.shape} for {layer_name}, skipping...")
                continue

            std = var.sqrt()

            stats[label_name] = {
                'mean': mean.cpu(),
                'var': var.cpu(),
                'std': std.cpu(),
            }

        separated_stats[layer_name] = stats

        # Print info
        C = len(stats['real']['mean'])
        real_mean_range = stats['real']['mean'].min().item(), stats['real']['mean'].max().item()
        fake_mean_range = stats['fake']['mean'].min().item(), stats['fake']['mean'].max().item()

        print(f"  {layer_name}: C={C}")
        print(f"    Real: mean=[{real_mean_range[0]:.4f}, {real_mean_range[1]:.4f}]")
        print(f"    Fake: mean=[{fake_mean_range[0]:.4f}, {fake_mean_range[1]:.4f}]")

        # Artifact signature
        artifact_sig = (stats['fake']['mean'] - stats['real']['mean']).abs()
        print(f"    Artifact signature: mean={artifact_sig.mean():.4f}, max={artifact_sig.max():.4f}")

    print(f"[CPv1] Separated statistics computed for {len(separated_stats)} layers")

    return separated_stats


class ArtifactAwareGating(nn.Module):
    """
    Gradient-Pattern-Aware Channel Gating Module (for LGrad)

    핵심 아이디어:
    - LGrad는 gradient pattern으로 deepfake 탐지
    - Noise가 gradient pattern을 corrupt
    - Gradient pattern을 robust하게 보존하는 채널 유지

    방법론 (IMPROVED):
    - Label-free minimum deviation으로 gradient corruption sensitivity 측정
    - Artifact discriminability (gradient pattern 차이) 보존
    - Score = discriminability / sensitivity
    - High score → keep (robust gradient pattern detection)
    - Low score → prune (noise-sensitive, poor discrimination)
    """

    def __init__(
        self,
        num_channels: int,
        layer_name: str,
        separated_stats: Dict[str, Dict[str, torch.Tensor]],
        config: CPv1Config,
    ):
        super().__init__()
        self.layer_name = layer_name
        self.separated_stats = separated_stats
        self.cfg = config

        # Pre-compute artifact discriminability
        # = Gradient pattern difference between Real/Fake
        artifact_signature = torch.abs(
            separated_stats['fake']['mean'] - separated_stats['real']['mean']
        )
        self.register_buffer('artifact_discriminability', artifact_signature)

        # Learnable parameters
        if config.use_learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(config.temperature_init))
        else:
            self.register_buffer('temperature', torch.tensor(config.temperature_init))

        if config.use_channel_bias:
            self.channel_bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_buffer('channel_bias', torch.zeros(num_channels))

    def compute_gradient_corruption_sensitivity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient corruption sensitivity (LABEL-FREE)

        핵심 통찰:
        - Clean gradient → features close to one reference (real or fake)
        - Corrupted gradient → features far from BOTH references
        - min(dev_real, dev_fake) captures corruption-induced deviation

        For LGrad:
        - Clean: min deviation is small (close to at least one reference)
        - Noisy: min deviation is large (far from both references)

        Args:
            x: [B, C, H, W] or [B, C] - features from classifier

        Returns:
            sensitivity: [B, C] - gradient corruption sensitivity
        """
        # Current statistics (from potentially corrupted gradient)
        if x.dim() == 4:
            curr_mean = x.mean(dim=[2, 3])  # [B, C]
            curr_var = x.var(dim=[2, 3]) if self.cfg.deviation_metric in ["mean+var", "var"] else None
        elif x.dim() == 2:
            curr_mean = x  # [B, C]
            curr_var = None  # No spatial dimension
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # Reference statistics (from clean gradients)
        real_mean = self.separated_stats['real']['mean'].to(x.device)  # [C]
        fake_mean = self.separated_stats['fake']['mean'].to(x.device)  # [C]

        # Compute deviation from each reference
        if self.cfg.deviation_metric == "mean+var" and curr_var is not None:
            real_var = self.separated_stats['real']['var'].to(x.device)
            fake_var = self.separated_stats['fake']['var'].to(x.device)

            if self.cfg.normalize_deviation:
                mean_dev_real = torch.abs(curr_mean - real_mean) / (real_mean.abs() + 1e-8)
                mean_dev_fake = torch.abs(curr_mean - fake_mean) / (fake_mean.abs() + 1e-8)
                var_dev_real = torch.abs(curr_var - real_var) / (real_var + 1e-8)
                var_dev_fake = torch.abs(curr_var - fake_var) / (fake_var + 1e-8)
            else:
                mean_dev_real = torch.abs(curr_mean - real_mean)
                mean_dev_fake = torch.abs(curr_mean - fake_mean)
                var_dev_real = torch.abs(curr_var - real_var)
                var_dev_fake = torch.abs(curr_var - fake_var)

            dev_real = mean_dev_real + var_dev_real  # [B, C]
            dev_fake = mean_dev_fake + var_dev_fake  # [B, C]

        elif self.cfg.deviation_metric == "mean":
            if self.cfg.normalize_deviation:
                dev_real = torch.abs(curr_mean - real_mean) / (real_mean.abs() + 1e-8)
                dev_fake = torch.abs(curr_mean - fake_mean) / (fake_mean.abs() + 1e-8)
            else:
                dev_real = torch.abs(curr_mean - real_mean)
                dev_fake = torch.abs(curr_mean - fake_mean)

        elif self.cfg.deviation_metric == "var" and curr_var is not None:
            real_var = self.separated_stats['real']['var'].to(x.device)
            fake_var = self.separated_stats['fake']['var'].to(x.device)

            if self.cfg.normalize_deviation:
                dev_real = torch.abs(curr_var - real_var) / (real_var + 1e-8)
                dev_fake = torch.abs(curr_var - fake_var) / (fake_var + 1e-8)
            else:
                dev_real = torch.abs(curr_var - real_var)
                dev_fake = torch.abs(curr_var - fake_var)
        else:
            # Fallback to mean if var is not available
            dev_real = torch.abs(curr_mean - real_mean)
            dev_fake = torch.abs(curr_mean - fake_mean)

        # Combine deviations based on sensitivity method
        if self.cfg.sensitivity_method == "min":
            # RECOMMENDED: Minimum deviation
            # Clean: close to at least one → small
            # Noisy: far from both → large
            sensitivity = torch.min(dev_real, dev_fake)  # [B, C]

        elif self.cfg.sensitivity_method == "avg":
            # Average deviation
            # Measures overall distance from references
            sensitivity = (dev_real + dev_fake) / 2  # [B, C]

        elif self.cfg.sensitivity_method == "max":
            # Maximum deviation (conservative)
            # Most sensitive to any deviation
            sensitivity = torch.max(dev_real, dev_fake)  # [B, C]

        else:
            raise ValueError(f"Unknown sensitivity_method: {self.cfg.sensitivity_method}")

        return sensitivity  # [B, C]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gradient-pattern-aware channel gating (IMPROVED)

        Args:
            x: [B, C, H, W] or [B, C]

        Returns:
            x_gated: [B, C, H, W] or [B, C] - gated output
            gate: [C] or [B, C] - gate values (for analysis)
        """
        # Step 1: Gradient corruption sensitivity (LABEL-FREE!)
        sensitivity = self.compute_gradient_corruption_sensitivity(x)  # [B, C]

        # Step 2: Artifact discriminability (pre-computed)
        # = Gradient pattern difference between Real/Fake
        disc = self.artifact_discriminability.unsqueeze(0)  # [1, C]

        # Step 3: Channel score
        # High score = Strong gradient pattern detection + Low corruption sensitivity
        epsilon = 1e-6
        score = disc / (sensitivity + epsilon)  # [B, C]

        # Step 4: Gating (with optional batch aggregation)
        if self.cfg.enable_batch_aggregation:
            # Aggregate across batch for channel-level gating (stable)
            if self.cfg.aggregation_method == "mean":
                score_agg = score.mean(dim=0)  # [C]
            elif self.cfg.aggregation_method == "median":
                score_agg = score.median(dim=0)[0]  # [C]
            elif self.cfg.aggregation_method == "min":
                score_agg = score.min(dim=0)[0]  # [C]
            else:
                raise ValueError(f"Unknown aggregation_method: {self.cfg.aggregation_method}")

            # Gating (channel-level)
            gate_logits = self.temperature * score_agg + self.channel_bias  # [C]

            if self.cfg.gating_type == "soft":
                gate = torch.sigmoid(gate_logits)  # [C]
            else:  # hard
                gate = (torch.sigmoid(gate_logits) > self.cfg.hard_threshold).float()

            # Broadcast to batch
            if x.dim() == 4:
                gate_broadcast = gate.view(1, -1, 1, 1)  # [1, C, 1, 1]
            elif x.dim() == 2:
                gate_broadcast = gate.unsqueeze(0)  # [1, C]
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")

        else:
            # Batch-wise gating (original CPv1 style)
            gate_logits = self.temperature * score + self.channel_bias.unsqueeze(0)  # [B, C]

            if self.cfg.gating_type == "soft":
                gate = torch.sigmoid(gate_logits)  # [B, C]
            else:  # hard
                gate = (torch.sigmoid(gate_logits) > self.cfg.hard_threshold).float()

            # Apply gating
            if x.dim() == 4:
                gate_broadcast = gate.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
            elif x.dim() == 2:
                gate_broadcast = gate  # [B, C]
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")

        # Apply gating
        x_gated = x * gate_broadcast

        return x_gated, gate


class UnifiedChannelPruningV1(nn.Module):
    """
    Unified Channel Pruning v1 for both LGrad and NPR (IMPROVED)

    핵심 (IMPROVED):
    - Real/Fake separated statistics 기반 (gradient pattern 차이)
    - Artifact discriminability 보존하면서 gradient corruption에 robust한 channel 유지
    - LABEL-FREE minimum deviation으로 sensitivity 측정 (pseudo-label 제거!)
    - Batch aggregation으로 안정적인 channel-level gating
    - Test-time adaptation 지원

    사용법:
        # 1. Separated statistics 수집 (Real/Fake 분리)
        stats = compute_separated_statistics(lgrad_model, clean_loader)

        # 2. Config 설정 (IMPROVED defaults)
        config = CPv1Config(
            model="LGrad",
            target_layers=['classifier.layer3', 'classifier.layer4'],
            sensitivity_method="min",  # RECOMMENDED: minimum deviation
            enable_batch_aggregation=True,  # Stable gating
            temperature_init=1.0
        )

        # 3. Model 생성
        model = UnifiedChannelPruningV1(lgrad_model, stats, config)

        # 4. Inference
        probs = model.predict_proba(noisy_images)
    """

    def __init__(
        self,
        base_model,
        separated_stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
        config: CPv1Config,
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

        print(f"[CPv1 IMPROVED] Initialized for {config.model}")
        print(f"[CPv1 IMPROVED] Target layers: {len(self.gates)}")
        print(f"[CPv1 IMPROVED] Sensitivity method: {config.sensitivity_method} (label-free!)")
        print(f"[CPv1 IMPROVED] Batch aggregation: {config.enable_batch_aggregation}")
        print(f"[CPv1 IMPROVED] Temperature init: {config.temperature_init} (learnable={config.use_learnable_temperature})")
        print(f"[CPv1 IMPROVED] Gating type: {config.gating_type}")

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
            gate_module = ArtifactAwareGating(
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

            # Print artifact discriminability info
            artifact_disc = gate_module.artifact_discriminability
            print(f"  {layer_name} (C={num_channels}):")
            print(f"    Artifact disc: mean={artifact_disc.mean():.4f}, max={artifact_disc.max():.4f}")

    def forward(
        self,
        x: torch.Tensor,
        return_artifact: bool = False,
        return_gates: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass with artifact-aware channel pruning

        Args:
            x: Input images [B, 3, H, W]
            return_artifact: If True, return artifact
            return_gates: If True, return gate values

        Returns:
            logits: [B, 1]
            artifact (optional)
            gates (optional): {layer_name: gate_values}
        """
        self.model.eval()

        # Forward (gates automatically applied via hooks)
        if self.cfg.model == "LGrad":
            if return_artifact:
                logits, artifact = self.model(x, return_grad=True)
            else:
                logits = self.model(x)
        else:  # NPR
            if return_artifact:
                logits, artifact = self.model(x, return_npr=True)
            else:
                logits = self.model(x)

        # Collect gates if requested
        if return_gates:
            # TODO: Implement proper gate collection
            gate_values = {}

        # Return
        returns = [logits]
        if return_artifact:
            returns.append(artifact)
        if return_gates:
            returns.append(gate_values)

        if len(returns) == 1:
            return returns[0]
        else:
            return tuple(returns)

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

    def adapt(
        self,
        dataloader,
        epochs: int = 5,
        lr: Optional[float] = None,
        verbose: bool = True,
    ):
        """
        Test-time adaptation on validation/test data

        Only optimizes gate parameters (temperature, channel_bias)
        """
        if not self.cfg.enable_adaptation:
            print("[CPv1] Adaptation disabled. Set enable_adaptation=True in config.")
            return

        if lr is None:
            lr = self.cfg.adaptation_lr

        # Collect gate parameters
        gate_params = []
        for gate in self.gates.values():
            gate_params.extend(list(gate.parameters()))

        optimizer = torch.optim.Adam(gate_params, lr=lr)

        if verbose:
            print(f"[CPv1] Starting adaptation for {epochs} epochs with lr={lr}")
            print(f"[CPv1] Optimizing {sum(p.numel() for p in gate_params)} parameters")

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)

                # Forward
                logits = self.forward(images)

                # Adaptation loss
                if self.cfg.adaptation_loss == "entropy":
                    probs = torch.sigmoid(logits)
                    entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
                    loss = entropy.mean()
                elif self.cfg.adaptation_loss == "consistency":
                    probs = torch.sigmoid(logits)
                    loss = probs.var()
                else:
                    raise ValueError(f"Unknown adaptation_loss: {self.cfg.adaptation_loss}")

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches

            if verbose:
                print(f"  Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.6f}")

        if verbose:
            print(f"[CPv1] Adaptation complete")


# Utility functions

def create_lgrad_cpv1(
    stylegan_weights: str,
    classifier_weights: str,
    separated_stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
    target_layers: Optional[list[str]] = None,
    temperature_init: float = 1.0,
    device: str = "cuda",
) -> UnifiedChannelPruningV1:
    """
    Create LGrad model with Channel Pruning v1

    Args:
        stylegan_weights: Path to StyleGAN weights
        classifier_weights: Path to classifier weights
        separated_stats: Pre-computed separated statistics
        target_layers: Layers to apply gating
        temperature_init: Initial temperature
        device: Device

    Returns:
        UnifiedChannelPruningV1 model

    Example:
        >>> # Compute separated statistics
        >>> stats = compute_separated_statistics(lgrad, clean_loader)
        >>>
        >>> # Create model
        >>> model = create_lgrad_cpv1(
        ...     stylegan_weights="...",
        ...     classifier_weights="...",
        ...     separated_stats=stats,
        ...     target_layers=['classifier.layer4']
        ... )
        >>> probs = model.predict_proba(noisy_images)
    """
    from model.LGrad.lgrad_model import LGrad

    # Create base model
    lgrad = LGrad(
        stylegan_weights=stylegan_weights,
        classifier_weights=classifier_weights,
        device=device,
        resize=256,
    )

    # Config
    config = CPv1Config(
        model="LGrad",
        target_layers=target_layers,
        temperature_init=temperature_init,
        device=device,
    )

    # Apply CPv1
    model = UnifiedChannelPruningV1(lgrad, separated_stats, config)

    return model


def create_npr_cpv1(
    weights: str,
    separated_stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
    target_layers: Optional[list[str]] = None,
    temperature_init: float = 1.0,
    device: str = "cuda",
) -> UnifiedChannelPruningV1:
    """
    Create NPR model with Channel Pruning v1

    Args:
        weights: Path to NPR weights
        separated_stats: Pre-computed separated statistics
        target_layers: Layers to apply gating
        temperature_init: Initial temperature
        device: Device

    Returns:
        UnifiedChannelPruningV1 model

    Example:
        >>> stats = compute_separated_statistics(npr, clean_loader)
        >>> model = create_npr_cpv1(weights="...", separated_stats=stats)
        >>> probs = model.predict_proba(noisy_images)
    """
    from model.NPR.npr_model import NPR

    # Create base model
    npr = NPR(
        weights=weights,
        device=device,
    )

    # Config
    config = CPv1Config(
        model="NPR",
        target_layers=target_layers,
        temperature_init=temperature_init,
        device=device,
    )

    # Apply CPv1
    model = UnifiedChannelPruningV1(npr, separated_stats, config)

    return model
