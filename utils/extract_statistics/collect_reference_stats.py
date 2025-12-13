"""
Phase 2 Preparation: Collect Reference Frequency Statistics

ProGAN의 original(clean) 데이터에서 gradient의 band-wise energy 통계를 수집하여
FreqNorm의 reference로 사용할 E_ref를 생성합니다.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model.LGrad.lgrad_model import LGrad
from utils.data.dataset import CorruptedDataset


# Frequency band definitions (same as visualize_frequency_analysis.py)
FREQUENCY_BANDS = [
    (0, 4, "DC+VeryLow"),
    (4, 16, "Low"),
    (16, 64, "Mid"),
    (64, 128, "High"),
    (128, 256, "VeryHigh"),
]


def create_radial_frequency_mask(shape, r_min, r_max, device="cuda"):
    """Create radial frequency mask for FFT."""
    H, W = shape
    cy, cx = H // 2, W // 2
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device) - cy,
        torch.arange(W, dtype=torch.float32, device=device) - cx,
        indexing='ij'
    )
    radius = torch.sqrt(x**2 + y**2)
    mask = (radius >= r_min) & (radius < r_max)
    return mask


def compute_band_energy(grad_fft_shifted, band_mask):
    """Compute energy in a frequency band."""
    power = torch.abs(grad_fft_shifted) ** 2

    if band_mask.sum() == 0:
        return 0.0

    energy = power[:, band_mask].mean().item()
    return energy


def collect_reference_statistics(
    dataset,
    lgrad_model,
    device="cuda",
    n_samples=1000,
    batch_size=16,
):
    """
    Collect reference frequency statistics from clean data.

    Args:
        dataset: CorruptedDataset with corruption="original" only
        lgrad_model: LGrad model
        device: computation device
        n_samples: number of samples to use (None = use all)
        batch_size: batch size for processing

    Returns:
        reference_stats: dict with band energies and statistics
    """
    print("="*80)
    print("Collecting Reference Frequency Statistics")
    print("="*80)

    # Sample dataset if needed
    if n_samples and n_samples < len(dataset):
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        dataset = Subset(dataset, indices)
        print(f"Using {n_samples} random samples from dataset")
    else:
        print(f"Using all {len(dataset)} samples from dataset")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    # Storage for band energies
    all_band_energies = [[] for _ in FREQUENCY_BANDS]

    # Create band masks (reusable)
    H, W = 256, 256
    band_masks = []
    for r_min, r_max, band_name in FREQUENCY_BANDS:
        mask = create_radial_frequency_mask((H, W), r_min, r_max, device=device)
        band_masks.append(mask)

    # Process batches
    lgrad_model.eval()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        images, labels, metadata = batch
        images = images.to(device)

        # Extract gradients
        grads = lgrad_model.img2grad(images)  # [B, 3, 256, 256]

        # Process each sample in batch
        for i in range(grads.shape[0]):
            grad = grads[i]  # [3, 256, 256]

            # FFT and shift
            grad_fft = torch.fft.fft2(grad)
            grad_fft_shifted = torch.fft.fftshift(grad_fft)

            # Compute band-wise energy
            for band_idx, mask in enumerate(band_masks):
                energy = compute_band_energy(grad_fft_shifted, mask)
                all_band_energies[band_idx].append(energy)

    # Compute statistics
    print("\n" + "="*80)
    print("Computing Statistics")
    print("="*80)

    reference_stats = {
        'bands': FREQUENCY_BANDS,
        'n_samples': len(all_band_energies[0]),
        'mean': [],
        'std': [],
        'median': [],
        'min': [],
        'max': [],
    }

    print(f"\n{'Band':<15} | {'Mean':>12} | {'Std':>12} | {'Median':>12} | {'Min':>12} | {'Max':>12}")
    print("-" * 90)

    for band_idx, (r_min, r_max, band_name) in enumerate(FREQUENCY_BANDS):
        energies = np.array(all_band_energies[band_idx])

        mean_val = np.mean(energies)
        std_val = np.std(energies)
        median_val = np.median(energies)
        min_val = np.min(energies)
        max_val = np.max(energies)

        reference_stats['mean'].append(mean_val)
        reference_stats['std'].append(std_val)
        reference_stats['median'].append(median_val)
        reference_stats['min'].append(min_val)
        reference_stats['max'].append(max_val)

        print(f"{band_name:<15} | {mean_val:>12.2e} | {std_val:>12.2e} | "
              f"{median_val:>12.2e} | {min_val:>12.2e} | {max_val:>12.2e}")

    # Convert to tensors for easy use
    reference_stats['mean'] = torch.tensor(reference_stats['mean'], dtype=torch.float32)
    reference_stats['std'] = torch.tensor(reference_stats['std'], dtype=torch.float32)
    reference_stats['median'] = torch.tensor(reference_stats['median'], dtype=torch.float32)

    print("\n" + "="*80)
    print("Statistics Collection Complete!")
    print("="*80)

    return reference_stats


def save_reference_stats(stats, output_path):
    """Save reference statistics to file."""
    torch.save(stats, output_path)
    print(f"\nReference statistics saved to: {output_path}")


def load_reference_stats(input_path):
    """Load reference statistics from file."""
    stats = torch.load(input_path, weights_only=False)
    print(f"Reference statistics loaded from: {input_path}")
    return stats


def main():
    """Main execution function"""

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load LGrad model
    print("Loading LGrad model...")
    lgrad = LGrad(
        stylegan_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/karras2019stylegan-bedrooms-256x256_discriminator.pth",
        classifier_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/LGrad-Pretrained-Model/LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth",
        device=device,
        resize=256,
    )
    lgrad.eval()
    print("Model loaded!\n")

    # Load ProGAN's original (clean) data only
    print("Loading ProGAN original (clean) dataset...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = CorruptedDataset(
        root="/workspace/robust_deepfake_ai/corrupted_dataset",
        datasets=["corrupted_test_data_progan"],  # ProGAN only
        corruptions=["original"],  # Original (clean) only
        transform=transform,
    )

    print(f"Dataset loaded: {len(dataset)} clean ProGAN samples\n")

    # Collect statistics
    reference_stats = collect_reference_statistics(
        dataset=dataset,
        lgrad_model=lgrad,
        device=device,
        n_samples=2000,  # Use 2000 samples for statistics (adjust as needed)
        batch_size=32,
    )

    # Save statistics
    output_path = "/workspace/robust_deepfake_ai/freq_reference_progan.pth"
    save_reference_stats(reference_stats, output_path)

    # Test loading
    print("\nTesting load...")
    loaded_stats = load_reference_stats(output_path)
    print(f"Loaded E_ref (mean): {loaded_stats['mean']}")

    print("\n" + "="*80)
    print("Reference Collection Complete!")
    print("="*80)
    print("\nReference: ProGAN original (clean) data")
    print(f"Sample size: {reference_stats['n_samples']}")
    print(f"Output: freq_reference_progan.pth")
    print("\nNext steps:")
    print("1. Use freq_reference_progan.pth for FreqNorm")
    print("2. Implement LGradFreqNorm class")
    print("3. Run evaluation on corrupted datasets")
    print("="*80)


if __name__ == "__main__":
    main()
