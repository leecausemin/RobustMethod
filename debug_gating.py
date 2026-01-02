"""
Gating 동작 분석 스크립트
"""
import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from utils.data.dataset import CorruptedDataset
from model.method import (
    UnifiedChannelReweightV1,
    ChannelReweightV1Config,
)
from model.LGrad.lgrad_model import LGrad

DEVICE = "cuda:0"
MODEL = "lgrad"

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Dataset
dataset = CorruptedDataset(
    root="corrupted_dataset",
    datasets=["corrupted_test_data_progan"],
    corruptions=["original", "gaussian_noise"],
    transform=transform
)

# Base model
base_model = LGrad(
    stylegan_weights="model/LGrad/weights/karras2019stylegan-bedrooms-256x256_discriminator.pth",
    classifier_weights="model/LGrad/weights/LGrad-Pretrained-Model/LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth",
    device=DEVICE
)

# Load clean stats
STATS_PATH = f"clean_stats_{MODEL}_progan_blocks.pth"
clean_stats = torch.load(STATS_PATH)

# Create model
config = ChannelReweightV1Config(
    model="LGrad",
    target_layers=[
        'classifier.layer1',
        'classifier.layer2',
        'classifier.layer3',
        'classifier.layer4',
    ],
    temperature_init=2.0,
    use_learnable_temperature=True,
    use_channel_bias=True,
    deviation_metric="mean+var",
    normalize_deviation=True,
    enable_adaptation=False,
    device=DEVICE,
)

model = UnifiedChannelReweightV1(
    base_model=base_model,
    clean_stats=clean_stats,
    config=config,
)

print("\n" + "="*80)
print("Gating Analysis")
print("="*80)

# Test on clean vs noisy images
for corruption in ["original", "gaussian_noise"]:
    print(f"\n### Testing on: {corruption}")

    # Get samples
    indices = [
        i for i, s in enumerate(dataset.samples)
        if s['dataset'] == "corrupted_test_data_progan" and s['corruption'] == corruption
    ][:32]  # 32 samples

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False)

    # Get one batch
    batch = next(iter(loader))
    images = batch[0].to(DEVICE)

    # Collect gate information
    gate_info = {}

    def make_analysis_hook(layer_name, gate_module):
        def hook(module, input, output):
            # Compute sensitivity and gate
            sensitivity = gate_module.compute_noise_sensitivity(output)
            gate_logits = -gate_module.temperature * sensitivity + gate_module.channel_bias
            gate_weights = torch.sigmoid(gate_logits)

            gate_info[layer_name] = {
                'sensitivity': sensitivity.detach().cpu(),
                'gate_weights': gate_weights.detach().cpu(),
                'temperature': gate_module.temperature.item(),
                'output_mean': output.mean().item(),
                'output_std': output.std().item(),
            }
        return hook

    # Register analysis hooks
    handles = []
    for sanitized_name, gate_module in model.gates.items():
        original_name = model.gate_name_mapping[sanitized_name]
        layer_module = dict(model.model.named_modules())[original_name]
        handle = layer_module.register_forward_hook(make_analysis_hook(original_name, gate_module))
        handles.append(handle)

    # Forward pass
    with torch.no_grad():
        try:
            _ = model(images)
        except:
            _ = model(images)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Print analysis
    for layer_name in ['classifier.layer1', 'classifier.layer2', 'classifier.layer3', 'classifier.layer4']:
        info = gate_info[layer_name]
        sens = info['sensitivity']
        gates = info['gate_weights']
        temp = info['temperature']

        print(f"\n  {layer_name}:")
        print(f"    Temperature: {temp:.4f}")
        print(f"    Sensitivity: min={sens.min():.4f}, max={sens.max():.4f}, mean={sens.mean():.4f}, median={sens.median():.4f}")
        print(f"    Gate weights: min={gates.min():.4f}, max={gates.max():.4f}, mean={gates.mean():.4f}, median={gates.median():.4f}")
        print(f"    Suppressed channels (gate < 0.5): {(gates < 0.5).sum().item()} / {len(gates)}")
        print(f"    Strong suppression (gate < 0.1): {(gates < 0.1).sum().item()} / {len(gates)}")
        print(f"    Output stats: mean={info['output_mean']:.4f}, std={info['output_std']:.4f}")

print("\n" + "="*80)
print("Analysis Complete")
print("="*80)
