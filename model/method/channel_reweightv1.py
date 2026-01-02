"""
Channel Reweight v1 (CRv1) - Noise-Robust Channel Gating

핵심 아이디어:
노이즈에 반응하는 channel의 가중치를 줄여서 robust하게 만들기

방법:
1. Clean data로 pre-train 시 각 layer의 channel별 statistics 저장
2. Test time에 current statistics와 비교하여 deviation 계산
3. Deviation이 큰 channel = noise-sensitive → 가중치 다운
4. Learnable temperature & channel-bias로 자동 튜닝

Based on:
- Statistical channel gating with clean data statistics
- Hybrid approach: statistical initialization + learnable refinement
- Test-time adaptation support
"""

import copy
from dataclasses import dataclass
from typing import Union, Optional, Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class ChannelReweightV1Config:
    """Configuration for Channel Reweight v1"""

    # Model type
    model: Literal["LGrad", "NPR"] = "LGrad"

    # Target layers to apply gating
    # Example: ['layer1', 'layer2', 'layer3', 'layer4'] for ResNet
    # None = automatically select all conv/bn layers
    target_layers: Optional[list[str]] = None

    # Gating parameters
    temperature_init: float = 2.0  # Initial temperature (learnable)
    use_learnable_temperature: bool = True  # Whether to make temperature learnable
    use_channel_bias: bool = True  # Whether to use channel-specific bias

    # Statistics computation
    deviation_metric: Literal["mean+var", "mean", "var"] = "mean+var"
    normalize_deviation: bool = True  # Normalize deviation by clean statistics

    # Test-time adaptation
    enable_adaptation: bool = False  # Whether to enable test-time adaptation
    adaptation_lr: float = 1e-4
    adaptation_loss: Literal["entropy", "consistency"] = "entropy"

    device: str = "cuda"

    def __post_init__(self):
        """Validate configuration"""
        if self.temperature_init <= 0:
            raise ValueError("temperature_init must be positive")


def compute_channel_statistics(
    model,
    dataloader,
    target_layers: Optional[list[str]] = None,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute channel-wise statistics from clean data

    Args:
        model: Pre-trained model (LGrad or NPR)
        dataloader: DataLoader with clean images
        target_layers: List of layer names to collect statistics
                      If None, collect from all Conv2d and BatchNorm2d layers
        device: Device to use
        max_batches: Maximum number of batches to process (for speed)

    Returns:
        stats: {layer_name: {'mean': [C], 'var': [C], 'std': [C]}}

    Example:
        >>> from torch.utils.data import DataLoader
        >>> clean_loader = DataLoader(clean_dataset, batch_size=32)
        >>> stats = compute_channel_statistics(lgrad_model, clean_loader)
    """
    model.eval()
    model.to(device)

    # Determine target layers
    if target_layers is None:
        target_layers = []
        for name, module in model.named_modules():
            # Only select layers from classifier (not from grad_model for LGrad)
            # This reduces memory usage significantly
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                # Skip grad_model layers for LGrad (they're not needed for gating)
                if 'grad_model' not in name:
                    target_layers.append(name)

    print(f"[CRv1] Computing statistics for {len(target_layers)} layers...")
    if len(target_layers) > 100:
        print(f"[CRv1] Warning: {len(target_layers)} layers detected. This may use a lot of memory.")
        print(f"[CRv1] Consider specifying target_layers explicitly to reduce memory usage.")

    # Storage for activations
    activations = {name: [] for name in target_layers}

    # Register hooks
    handles = []

    def make_hook(layer_name):
        def hook(module, input, output):
            # output: [B, C, H, W] or [B, C]
            # Immediately move to CPU and detach to save GPU memory
            if output.dim() == 4:
                # Spatial features: [B, C, H, W]
                activations[layer_name].append(output.detach().cpu())
            elif output.dim() == 2:
                # Fully connected: [B, C]
                activations[layer_name].append(output.detach().cpu())
        return hook

    for name in target_layers:
        module = dict(model.named_modules())[name]
        handle = module.register_forward_hook(make_hook(name))
        handles.append(handle)

    # Collect activations
    total_batches = max_batches if max_batches is not None else len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=total_batches, desc="Computing statistics")

    for batch_idx, batch in pbar:
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Handle different batch formats
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch

        images = images.to(device)

        # Forward pass with no gradient
        # Note: LGrad internally uses autograd.grad, but we don't keep the graph
        try:
            with torch.no_grad():
                _ = model(images)
        except:
            # LGrad might need gradients for img2grad
            # So we allow it but clear immediately
            try:
                _ = model(images)
            except:
                # Some models might have different interfaces
                if hasattr(model, 'model'):
                    _ = model.model(images)
                else:
                    raise

        # Clear GPU memory after each batch
        del images
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

        # Update progress bar
        pbar.set_postfix({'batch': f'{batch_idx + 1}/{total_batches}'})

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Compute statistics
    stats = {}

    for layer_name in target_layers:
        if len(activations[layer_name]) == 0:
            print(f"  Warning: No activations collected for {layer_name}, skipping...")
            continue

        # Concatenate all batches: [Total_B, C, H, W] or [Total_B, C]
        all_acts = torch.cat(activations[layer_name], dim=0)

        # Compute channel-wise statistics
        if all_acts.dim() == 4:
            # Spatial: average over batch, height, width → [C]
            channel_mean = all_acts.mean(dim=[0, 2, 3])
            channel_var = all_acts.var(dim=[0, 2, 3])
        elif all_acts.dim() == 2:
            # Fully connected: average over batch → [C]
            channel_mean = all_acts.mean(dim=0)
            channel_var = all_acts.var(dim=0)
        else:
            print(f"  Warning: Unexpected activation shape {all_acts.shape} for {layer_name}, skipping...")
            continue

        channel_std = channel_var.sqrt()

        stats[layer_name] = {
            'mean': channel_mean.cpu(),
            'var': channel_var.cpu(),
            'std': channel_std.cpu(),
        }

        print(f"  {layer_name}: C={len(channel_mean)}, "
              f"mean_range=[{channel_mean.min():.4f}, {channel_mean.max():.4f}], "
              f"var_range=[{channel_var.min():.4f}, {channel_var.max():.4f}]")

    print(f"[CRv1] Statistics computed for {len(stats)} layers")

    return stats


class HybridChannelGating(nn.Module):
    """
    Hybrid Channel Gating Module

    - Compute noise sensitivity from clean statistics
    - Apply learnable gating: high sensitivity → low gate value
    - Temperature and channel bias are learnable
    """

    def __init__(
        self,
        num_channels: int,
        layer_name: str,
        clean_stats: Dict[str, torch.Tensor],
        config: ChannelReweightV1Config,
    ):
        super().__init__()
        self.layer_name = layer_name
        self.clean_stats = clean_stats
        self.cfg = config

        # Learnable parameters
        if config.use_learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(config.temperature_init))
        else:
            self.register_buffer('temperature', torch.tensor(config.temperature_init))

        if config.use_channel_bias:
            self.channel_bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_buffer('channel_bias', torch.zeros(num_channels))

    def compute_noise_sensitivity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute channel-wise noise sensitivity

        Args:
            x: [B, C, H, W] or [B, C]

        Returns:
            sensitivity: [C] - higher value = more noise-sensitive
        """
        # Current statistics
        if x.dim() == 4:
            curr_mean = x.mean(dim=[0, 2, 3])  # [C]
            curr_var = x.var(dim=[0, 2, 3])    # [C]
        elif x.dim() == 2:
            curr_mean = x.mean(dim=0)  # [C]
            curr_var = x.var(dim=0)    # [C]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # Clean reference statistics
        clean_mean = self.clean_stats['mean'].to(x.device)
        clean_var = self.clean_stats['var'].to(x.device)

        # Compute deviation
        if self.cfg.deviation_metric == "mean+var":
            # Both mean and variance deviation
            if self.cfg.normalize_deviation:
                mean_dev = torch.abs(curr_mean - clean_mean) / (clean_mean.abs() + 1e-8)
                var_dev = torch.abs(curr_var - clean_var) / (clean_var + 1e-8)
            else:
                mean_dev = torch.abs(curr_mean - clean_mean)
                var_dev = torch.abs(curr_var - clean_var)

            sensitivity = mean_dev + var_dev

        elif self.cfg.deviation_metric == "mean":
            if self.cfg.normalize_deviation:
                sensitivity = torch.abs(curr_mean - clean_mean) / (clean_mean.abs() + 1e-8)
            else:
                sensitivity = torch.abs(curr_mean - clean_mean)

        elif self.cfg.deviation_metric == "var":
            if self.cfg.normalize_deviation:
                sensitivity = torch.abs(curr_var - clean_var) / (clean_var + 1e-8)
            else:
                sensitivity = torch.abs(curr_var - clean_var)

        else:
            raise ValueError(f"Unknown deviation_metric: {self.cfg.deviation_metric}")

        return sensitivity

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply channel gating

        Args:
            x: [B, C, H, W] or [B, C]

        Returns:
            x_gated: [B, C, H, W] or [B, C] - gated output
            gate: [C] - gate values (for visualization/analysis)
        """
        # Compute noise sensitivity
        sensitivity = self.compute_noise_sensitivity(x)  # [C]

        # Apply learnable gating
        # High sensitivity → negative logit → low gate value (closer to 0)
        # Low sensitivity → positive logit → high gate value (closer to 1)
        gate_logits = -self.temperature * sensitivity + self.channel_bias
        gate = torch.sigmoid(gate_logits)  # [C]

        # Apply gating with broadcasting
        if x.dim() == 4:
            gate_broadcast = gate.view(1, -1, 1, 1)  # [1, C, 1, 1]
        elif x.dim() == 2:
            gate_broadcast = gate.view(1, -1)  # [1, C]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        x_gated = x * gate_broadcast

        return x_gated, gate


class UnifiedChannelReweightV1(nn.Module):
    """
    Unified Channel Reweight v1 for both LGrad and NPR

    핵심:
    - Clean data statistics 기반 channel gating
    - Noise-sensitive channels을 다운웨이팅
    - Learnable temperature & bias로 자동 튜닝
    - Test-time adaptation 지원

    사용법:
        # 1. Clean data로 statistics 수집
        stats = compute_channel_statistics(lgrad_model, clean_loader)

        # 2. Config 설정
        config = ChannelReweightV1Config(
            model="LGrad",
            target_layers=['classifier.layer3', 'classifier.layer4'],
            temperature_init=2.0
        )

        # 3. Model 생성
        model = UnifiedChannelReweightV1(lgrad_model, stats, config)

        # 4. (Optional) Test-time adaptation
        model.adapt(noisy_validation_loader, epochs=5)

        # 5. Inference
        probs = model.predict_proba(test_images)
    """

    def __init__(
        self,
        base_model,
        clean_stats: Dict[str, Dict[str, torch.Tensor]],
        config: ChannelReweightV1Config,
    ):
        super().__init__()
        self.cfg = config
        self.device = config.device

        # Deep copy to avoid modifying original model
        self.model = copy.deepcopy(base_model)
        self.model.to(self.device)

        # Validate config
        if config.model not in ["LGrad", "NPR"]:
            raise ValueError(f"Unsupported model type: {config.model}")

        # Store clean statistics
        self.clean_stats = clean_stats

        # Setup gating modules
        self.gates = nn.ModuleDict()
        # Mapping from sanitized names (used in ModuleDict) to original layer names
        self.gate_name_mapping = {}
        self._setup_gates()

        print(f"[CRv1] Initialized for {config.model}")
        print(f"[CRv1] Target layers: {len(self.gates)}")
        print(f"[CRv1] Temperature init: {config.temperature_init} (learnable={config.use_learnable_temperature})")
        print(f"[CRv1] Channel bias: {config.use_channel_bias}")
        print(f"[CRv1] Deviation metric: {config.deviation_metric}")

    def _setup_gates(self):
        """Setup gating modules for target layers"""

        # Determine target layers
        target_layers = self.cfg.target_layers
        if target_layers is None:
            # Use all layers in clean_stats
            target_layers = list(self.clean_stats.keys())

        # Validate that target layers exist in stats
        for layer_name in target_layers:
            if layer_name not in self.clean_stats:
                raise ValueError(f"Layer {layer_name} not found in clean_stats. "
                               f"Available layers: {list(self.clean_stats.keys())}")

        # Create gating modules and install hooks
        for layer_name in target_layers:
            # Get layer module
            try:
                layer_module = dict(self.model.named_modules())[layer_name]
            except KeyError:
                print(f"  Warning: Layer {layer_name} not found in model, skipping...")
                continue

            # Determine number of channels from clean_stats
            # (Sequential modules don't have out_channels, so we get it from stats)
            num_channels = len(self.clean_stats[layer_name]['mean'])

            # Optional: verify with module attributes if available
            if hasattr(layer_module, 'out_channels'):
                if layer_module.out_channels != num_channels:
                    print(f"  Warning: Channel mismatch for {layer_name}: "
                          f"model={layer_module.out_channels}, stats={num_channels}, using stats={num_channels}")
            elif hasattr(layer_module, 'num_features'):
                if layer_module.num_features != num_channels:
                    print(f"  Warning: Channel mismatch for {layer_name}: "
                          f"model={layer_module.num_features}, stats={num_channels}, using stats={num_channels}")
            elif hasattr(layer_module, 'out_features'):
                if layer_module.out_features != num_channels:
                    print(f"  Warning: Channel mismatch for {layer_name}: "
                          f"model={layer_module.out_features}, stats={num_channels}, using stats={num_channels}")

            # Create gating module
            gate_module = HybridChannelGating(
                num_channels=num_channels,
                layer_name=layer_name,
                clean_stats=self.clean_stats[layer_name],
                config=self.cfg,
            )

            # Move gate module to device
            gate_module = gate_module.to(self.device)

            # Sanitize layer name for ModuleDict (replace . with _)
            sanitized_name = layer_name.replace('.', '_')
            self.gates[sanitized_name] = gate_module
            self.gate_name_mapping[sanitized_name] = layer_name

            # Install forward hook
            def make_hook(gate):
                def hook(module, input, output):
                    gated_output, gate_weights = gate(output)
                    return gated_output
                return hook

            layer_module.register_forward_hook(make_hook(gate_module))

            print(f"  Installed gate at {layer_name} (C={num_channels})")

    def forward(
        self,
        x: torch.Tensor,
        return_artifact: bool = False,
        return_gates: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass with channel gating

        Args:
            x: Input images [B, 3, H, W], range [0, 1]
            return_artifact: If True, also return artifact
            return_gates: If True, also return gate values for analysis

        Returns:
            logits: [B, 1] (positive = fake)
            artifact (optional): Artifact (gradient for LGrad, NPR for NPR)
            gates (optional): {layer_name: gate_values}
        """
        self.model.eval()

        # Forward through model (gates are automatically applied via hooks)
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

        # Collect gate values if requested
        if return_gates:
            # Re-run forward to collect gate values
            # (This is a bit inefficient but ensures correctness)
            gate_values = {}
            # TODO: Implement proper gate value collection
            # For now, return empty dict
            gate_values = {}

        # Return based on flags
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
        """
        Predict real/fake labels

        Args:
            x: Input images [B, 3, H, W], range [0, 1]

        Returns:
            predictions: [B] (0=real, 1=fake)
        """
        self.model.eval()
        logits = self.forward(x)
        return (torch.sigmoid(logits) > 0.5).long().squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probability of being fake

        Args:
            x: Input images [B, 3, H, W], range [0, 1]

        Returns:
            probabilities: [B] (probability of fake)
        """
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

        Args:
            dataloader: DataLoader with noisy images
            epochs: Number of adaptation epochs
            lr: Learning rate (default: use config.adaptation_lr)
            verbose: Whether to print progress
        """
        if not self.cfg.enable_adaptation:
            print("[CRv1] Adaptation is disabled in config. Set enable_adaptation=True to use.")
            return

        if lr is None:
            lr = self.cfg.adaptation_lr

        # Only optimize gate parameters
        gate_params = []
        for gate in self.gates.values():
            gate_params.extend(list(gate.parameters()))

        optimizer = torch.optim.Adam(gate_params, lr=lr)

        if verbose:
            print(f"[CRv1] Starting adaptation for {epochs} epochs with lr={lr}")
            print(f"[CRv1] Optimizing {sum(p.numel() for p in gate_params)} parameters")

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)

                # Forward pass
                logits = self.forward(images)

                # Compute adaptation loss
                if self.cfg.adaptation_loss == "entropy":
                    # Entropy minimization: encourage confident predictions
                    probs = torch.sigmoid(logits)
                    # Binary entropy: -p*log(p) - (1-p)*log(1-p)
                    entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
                    loss = entropy.mean()

                elif self.cfg.adaptation_loss == "consistency":
                    # Consistency loss: minimize prediction variance across batch
                    probs = torch.sigmoid(logits)
                    loss = probs.var()

                else:
                    raise ValueError(f"Unknown adaptation_loss: {self.cfg.adaptation_loss}")

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches

            if verbose:
                print(f"  Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.6f}")

        if verbose:
            print(f"[CRv1] Adaptation complete")
            # Print learned temperatures
            for sanitized_name, gate in self.gates.items():
                original_name = self.gate_name_mapping[sanitized_name]
                temp = gate.temperature.item() if isinstance(gate.temperature, nn.Parameter) else gate.temperature.item()
                print(f"  {original_name}: temperature={temp:.4f}")


# Utility functions for easy usage

def create_lgrad_crv1(
    stylegan_weights: str,
    classifier_weights: str,
    clean_stats: Dict[str, Dict[str, torch.Tensor]],
    target_layers: Optional[list[str]] = None,
    temperature_init: float = 2.0,
    device: str = "cuda",
) -> UnifiedChannelReweightV1:
    """
    Convenience function to create LGrad model with Channel Reweight v1

    Args:
        stylegan_weights: Path to StyleGAN discriminator weights
        classifier_weights: Path to ResNet50 classifier weights
        clean_stats: Pre-computed channel statistics from clean data
        target_layers: List of layer names to apply gating (None = auto)
        temperature_init: Initial temperature value
        device: Device to use

    Returns:
        UnifiedChannelReweightV1 model ready for inference

    Example:
        >>> # Step 1: Compute statistics
        >>> from model.LGrad.lgrad_model import LGrad
        >>> lgrad = LGrad(stylegan_weights="...", classifier_weights="...")
        >>> stats = compute_channel_statistics(lgrad, clean_loader)
        >>>
        >>> # Step 2: Create model with gating
        >>> model = create_lgrad_crv1(
        ...     stylegan_weights="...",
        ...     classifier_weights="...",
        ...     clean_stats=stats,
        ...     target_layers=['classifier.layer3', 'classifier.layer4']
        ... )
        >>> probs = model.predict_proba(corrupted_images)
    """
    from model.LGrad.lgrad_model import LGrad

    # Create base LGrad model
    lgrad = LGrad(
        stylegan_weights=stylegan_weights,
        classifier_weights=classifier_weights,
        device=device,
        resize=256,
    )

    # Apply Channel Reweight v1
    config = ChannelReweightV1Config(
        model="LGrad",
        target_layers=target_layers,
        temperature_init=temperature_init,
        device=device,
    )

    model = UnifiedChannelReweightV1(lgrad, clean_stats, config)

    return model


def create_npr_crv1(
    weights: str,
    clean_stats: Dict[str, Dict[str, torch.Tensor]],
    target_layers: Optional[list[str]] = None,
    temperature_init: float = 2.0,
    device: str = "cuda",
) -> UnifiedChannelReweightV1:
    """
    Convenience function to create NPR model with Channel Reweight v1

    Args:
        weights: Path to NPR model weights
        clean_stats: Pre-computed channel statistics from clean data
        target_layers: List of layer names to apply gating (None = auto)
        temperature_init: Initial temperature value
        device: Device to use

    Returns:
        UnifiedChannelReweightV1 model ready for inference

    Example:
        >>> # Step 1: Compute statistics
        >>> from model.NPR.npr_model import NPR
        >>> npr = NPR(weights="...")
        >>> stats = compute_channel_statistics(npr, clean_loader)
        >>>
        >>> # Step 2: Create model with gating
        >>> model = create_npr_crv1(
        ...     weights="...",
        ...     clean_stats=stats,
        ...     target_layers=['model.layer3', 'model.layer4']
        ... )
        >>> probs = model.predict_proba(corrupted_images)
    """
    from model.NPR.npr_model import NPR

    # Create base NPR model
    npr = NPR(
        weights=weights,
        device=device,
    )

    # Apply Channel Reweight v1
    config = ChannelReweightV1Config(
        model="NPR",
        target_layers=target_layers,
        temperature_init=temperature_init,
        device=device,
    )

    model = UnifiedChannelReweightV1(npr, clean_stats, config)

    return model
