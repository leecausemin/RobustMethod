"""
Phase 1: Frequency Analysis for FreqNorm Validation

이 스크립트는 다음을 검증합니다:
1. Corruption이 gradient의 주파수 스펙트럼을 왜곡하는가?
2. Band-wise energy 비율이 corruption에 따라 변하는가?
3. JPEG/blur는 고주파 감소, noise는 고주파 증가하는가?
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision import transforms
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.LGrad.lgrad_model import LGrad
from utils.data.dataset import CorruptedDataset


# Frequency band definitions (logarithmic split)
FREQUENCY_BANDS = [
    (0, 4, "DC+VeryLow"),
    (4, 16, "Low"),
    (16, 64, "Mid"),
    (64, 128, "High"),
    (128, 256, "VeryHigh"),
]


def create_radial_frequency_mask(shape, r_min, r_max):
    """
    Create a radial frequency mask for FFT.

    Args:
        shape: (H, W) of the frequency domain
        r_min: minimum radius
        r_max: maximum radius

    Returns:
        mask: boolean tensor [H, W]
    """
    H, W = shape
    # Create coordinate grids centered at (0, 0) after fftshift
    cy, cx = H // 2, W // 2
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32) - cy,
        torch.arange(W, dtype=torch.float32) - cx,
        indexing='ij'
    )

    # Radial distance from center
    radius = torch.sqrt(x**2 + y**2)

    # Create mask for this band
    mask = (radius >= r_min) & (radius < r_max)

    return mask


def compute_band_energy(grad_fft_shifted, band_mask):
    """
    Compute energy in a frequency band.

    Args:
        grad_fft_shifted: [3, H, W] complex tensor (after fftshift)
        band_mask: [H, W] boolean mask

    Returns:
        energy: scalar (mean power in this band)
    """
    # Power spectrum (magnitude squared)
    power = torch.abs(grad_fft_shifted) ** 2

    # Average over the band (across all channels)
    if band_mask.sum() == 0:
        return 0.0

    energy = power[:, band_mask].mean().item()
    return energy


def analyze_gradient_frequency(grad, device="cuda"):
    """
    Analyze frequency spectrum of gradient.

    Args:
        grad: [3, H, W] gradient tensor

    Returns:
        spectrum: [3, H, W] power spectrum (for visualization)
        band_energies: list of energies for each band
    """
    grad = grad.to(device)

    # 2D FFT
    grad_fft = torch.fft.fft2(grad)

    # Shift zero frequency to center (for visualization and radial mask)
    grad_fft_shifted = torch.fft.fftshift(grad_fft)

    # Power spectrum
    power_spectrum = torch.abs(grad_fft_shifted) ** 2

    # Log scale for visualization
    spectrum_vis = torch.log10(power_spectrum + 1e-10)

    # Compute band-wise energy
    band_energies = []
    H, W = grad.shape[1], grad.shape[2]

    for r_min, r_max, band_name in FREQUENCY_BANDS:
        mask = create_radial_frequency_mask((H, W), r_min, r_max)
        mask = mask.to(device)
        energy = compute_band_energy(grad_fft_shifted, mask)
        band_energies.append(energy)

    return spectrum_vis.cpu(), band_energies


def visualize_frequency_analysis(
    dataset,
    lgrad_model,
    corruptions=None,
    dataset_name=None,
    device="cuda"
):
    """
    Visualize frequency analysis for different corruptions.

    Args:
        dataset: CorruptedDataset
        lgrad_model: LGrad model
        corruptions: list of corruption types to analyze
        dataset_name: specific dataset to use (e.g., "corrupted_test_data_progan")
        device: computation device
    """
    if corruptions is None:
        corruptions = CorruptedDataset.CORRUPTIONS

    # Select one sample per corruption (same image across corruptions)
    samples_by_corruption = {}

    # Find samples that exist across all corruptions
    from collections import defaultdict
    filename_map = defaultdict(dict)

    for sample in dataset.samples:
        if dataset_name and sample["dataset"] != dataset_name:
            continue

        key = (sample["dataset"], sample["label"], sample["filename"])
        filename_map[key][sample["corruption"]] = sample

    # Find a complete sample (has all corruptions)
    for key, corr_dict in filename_map.items():
        if len(corr_dict) == len(corruptions):
            samples_by_corruption = corr_dict
            break

    if not samples_by_corruption:
        print("No complete sample found with all corruptions.")
        return

    print(f"Analyzing sample: {key[2]} from {key[0]}")

    # Prepare figure
    n_corruptions = len(corruptions)
    fig = plt.figure(figsize=(20, 4 * n_corruptions))
    gs = GridSpec(n_corruptions, 4, figure=fig, wspace=0.3, hspace=0.4)

    # Store band energies for comparison
    all_band_energies = {}

    # Process each corruption
    for idx, corruption in enumerate(corruptions):
        if corruption not in samples_by_corruption:
            continue

        sample_info = samples_by_corruption[corruption]

        # Load image
        img = Image.open(sample_info["path"]).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Extract gradient
        print(f"Processing {corruption}...")
        grad = lgrad_model.img2grad(img_tensor)

        grad = grad.squeeze(0)  # [3, 256, 256]

        # Frequency analysis
        spectrum, band_energies = analyze_gradient_frequency(grad, device)
        all_band_energies[corruption] = band_energies

        # Visualizations
        # 1. Original Image
        ax1 = fig.add_subplot(gs[idx, 0])
        img_np = img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        ax1.imshow(img_np)
        ax1.set_title(f"{corruption.replace('_', ' ').title()}\nOriginal Image",
                     fontsize=12, fontweight='bold')
        ax1.axis('off')

        # 2. Gradient Map (average across RGB)
        ax2 = fig.add_subplot(gs[idx, 1])
        grad_vis = grad.cpu().mean(dim=0).numpy()  # Average RGB for visualization
        ax2.imshow(grad_vis, cmap='gray')
        ax2.set_title("Gradient Map\n(RGB averaged)", fontsize=12, fontweight='bold')
        ax2.axis('off')

        # 3. Frequency Spectrum (log scale)
        ax3 = fig.add_subplot(gs[idx, 2])
        spectrum_vis = spectrum.mean(dim=0).numpy()  # Average RGB
        im = ax3.imshow(spectrum_vis, cmap='viridis', aspect='auto')
        ax3.set_title("Frequency Spectrum\n(log10 scale)", fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046)

        # 4. Band-wise Energy
        ax4 = fig.add_subplot(gs[idx, 3])
        band_names = [name for _, _, name in FREQUENCY_BANDS]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(band_names)))

        bars = ax4.bar(band_names, band_energies, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel("Energy (power)", fontsize=11)
        ax4.set_title("Band-wise Energy", fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, energy in zip(bars, band_energies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{energy:.2e}',
                    ha='center', va='bottom', fontsize=8)

    plt.suptitle(f"Frequency Analysis: {key[0]}", fontsize=16, fontweight='bold', y=0.995)

    # Save
    output_path = "/workspace/robust_deepfake_ai/frequency_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.show()

    # Print band energy comparison
    print("\n" + "="*80)
    print("Band-wise Energy Comparison")
    print("="*80)
    print(f"{'Corruption':<20} | " + " | ".join([f"{name:>12}" for _, _, name in FREQUENCY_BANDS]))
    print("-" * 80)

    for corruption in corruptions:
        if corruption in all_band_energies:
            energies = all_band_energies[corruption]
            print(f"{corruption:<20} | " + " | ".join([f"{e:>12.2e}" for e in energies]))

    # Compute normalized ratios (relative to 'original')
    if 'original' in all_band_energies:
        print("\n" + "="*80)
        print("Normalized Energy Ratios (relative to 'original')")
        print("="*80)
        print(f"{'Corruption':<20} | " + " | ".join([f"{name:>12}" for _, _, name in FREQUENCY_BANDS]))
        print("-" * 80)

        original_energies = np.array(all_band_energies['original'])

        for corruption in corruptions:
            if corruption in all_band_energies and corruption != 'original':
                energies = np.array(all_band_energies[corruption])
                ratios = energies / (original_energies + 1e-10)
                print(f"{corruption:<20} | " + " | ".join([f"{r:>12.3f}" for r in ratios]))

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)

    return all_band_energies


def plot_energy_comparison(all_band_energies, save_path="frequency_comparison.png"):
    """
    Create a comparative plot of band energies across corruptions.

    Args:
        all_band_energies: dict[corruption] -> list of band energies
        save_path: output path
    """
    corruptions = list(all_band_energies.keys())
    band_names = [name for _, _, name in FREQUENCY_BANDS]
    n_bands = len(FREQUENCY_BANDS)

    # Prepare data
    energies_matrix = np.array([all_band_energies[c] for c in corruptions])

    # Normalize by 'original' if available
    if 'original' in corruptions:
        original_idx = corruptions.index('original')
        original_energies = energies_matrix[original_idx]
        energies_normalized = energies_matrix / (original_energies + 1e-10)
    else:
        energies_normalized = energies_matrix

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Absolute energies
    ax1 = axes[0]
    x = np.arange(len(corruptions))
    width = 0.15

    for i, band_name in enumerate(band_names):
        offset = width * (i - n_bands/2)
        ax1.bar(x + offset, energies_matrix[:, i], width,
               label=band_name, alpha=0.8)

    ax1.set_xlabel("Corruption Type", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Energy (log scale)", fontsize=12, fontweight='bold')
    ax1.set_title("Absolute Band Energies", fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(corruptions, rotation=45, ha='right')
    ax1.set_yscale('log')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    # Right: Normalized ratios
    if 'original' in corruptions:
        ax2 = axes[1]

        for i, band_name in enumerate(band_names):
            offset = width * (i - n_bands/2)
            ax2.bar(x + offset, energies_normalized[:, i], width,
                   label=band_name, alpha=0.8)

        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Original')
        ax2.set_xlabel("Corruption Type", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Energy Ratio (relative to original)", fontsize=12, fontweight='bold')
        ax2.set_title("Normalized Band Energy Ratios", fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(corruptions, rotation=45, ha='right')
        ax2.legend(loc='upper right')
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    plt.show()


def main():
    """Main execution function"""

    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # LGrad 모델 로드
    print("Loading LGrad model...")
    lgrad = LGrad(
        stylegan_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/karras2019stylegan-bedrooms-256x256_discriminator.pth",
        classifier_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/LGrad-Pretrained-Model/LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth",
        device=device,
        resize=256,
    )
    lgrad.eval()
    print("Model loaded!\n")

    # 데이터셋 로드
    print("Loading dataset...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = CorruptedDataset(
        root="/workspace/robust_deepfake_ai/corrupted_dataset",
        datasets=["corrupted_test_data_progan"],  # Use ProGAN for speed
        transform=transform,
    )
    print(f"Dataset loaded: {len(dataset)} samples\n")

    # Frequency analysis
    corruptions = CorruptedDataset.CORRUPTIONS
    print(f"Analyzing corruptions: {corruptions}\n")

    all_band_energies = visualize_frequency_analysis(
        dataset=dataset,
        lgrad_model=lgrad,
        corruptions=corruptions,
        dataset_name="corrupted_test_data_progan",
        device=device,
    )

    # Additional comparison plot
    if all_band_energies:
        print("\nCreating comparison plot...")
        plot_energy_comparison(
            all_band_energies,
            save_path="/workspace/robust_deepfake_ai/frequency_comparison.png"
        )

    print("\n" + "="*80)
    print("Phase 1 Validation Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review frequency_analysis.png - Do corruptions change spectrum?")
    print("2. Review frequency_comparison.png - Do band ratios differ?")
    print("3. Check if:")
    print("   - JPEG/blur reduce high-frequency energy")
    print("   - Noise increases high-frequency energy")
    print("   - Original has balanced distribution")
    print("\nIf hypothesis is confirmed → Proceed to Phase 2 (FreqNorm implementation)")
    print("="*80)


if __name__ == "__main__":
    main()
