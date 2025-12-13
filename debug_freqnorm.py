"""
Debug FreqNorm: Visualize what FreqNorm is doing to gradients
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from model.LGrad.lgrad_model import LGrad
from model.method.freq_norm import LGradFreqNorm, create_lgrad_freqnorm
from utils.data.dataset import CorruptedDataset


def analyze_freqnorm_effect(device="cuda"):
    """Analyze what FreqNorm does to gradients"""

    print("Loading models...")

    # Original LGrad
    lgrad_original = LGrad(
        stylegan_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/karras2019stylegan-bedrooms-256x256_discriminator.pth",
        classifier_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/LGrad-Pretrained-Model/LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth",
        device=device,
        resize=256,
    )
    lgrad_original.eval()

    # FreqNorm model
    freqnorm_model = create_lgrad_freqnorm(
        stylegan_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/karras2019stylegan-bedrooms-256x256_discriminator.pth",
        classifier_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/LGrad-Pretrained-Model/LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth",
        reference_stats_path="/workspace/robust_deepfake_ai/freq_reference_progan.pth",
        rho=0.5,
        alpha_min=0.5,
        alpha_max=2.0,
        device=device
    )
    freqnorm_model.eval()

    print("Models loaded!\n")

    # Load test images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = CorruptedDataset(
        root="/workspace/robust_deepfake_ai/corrupted_dataset",
        datasets=["corrupted_test_data_progan"],
        corruptions=["gaussian_noise", "motion_blur", "original"],
        transform=transform,
    )

    # Test on different corruptions
    corruptions_to_test = ["original", "gaussian_noise", "motion_blur"]

    for corruption in corruptions_to_test:
        print(f"\n{'='*80}")
        print(f"Testing: {corruption}")
        print(f"{'='*80}")

        # Find sample
        sample = None
        for s in dataset.samples:
            if s["corruption"] == corruption and s["label"] == 1:  # fake
                sample = s
                break

        if not sample:
            print(f"No sample found for {corruption}")
            continue

        # Load image
        img = Image.open(sample["path"]).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Get original gradient
        grad_original = lgrad_original.img2grad(img_tensor)

        # Get FreqNorm gradient
        grad_freqnorm = freqnorm_model.img2grad_with_freqnorm(img_tensor)

        # Classify
        logits_original = lgrad_original.classify(grad_original)
        logits_freqnorm = freqnorm_model.lgrad.classify(grad_freqnorm)

        prob_original = torch.sigmoid(logits_original).item()
        prob_freqnorm = torch.sigmoid(logits_freqnorm).item()

        print(f"\nPredictions:")
        print(f"  Original LGrad:  {prob_original:.4f}")
        print(f"  FreqNorm:        {prob_freqnorm:.4f}")
        print(f"  Difference:      {prob_freqnorm - prob_original:+.4f}")

        # Compute statistics
        grad_orig_np = grad_original.squeeze(0).cpu().detach().numpy()
        grad_freq_np = grad_freqnorm.squeeze(0).cpu().detach().numpy()

        print(f"\nGradient Statistics:")
        print(f"  Original - Mean: {grad_orig_np.mean():.6f}, Std: {grad_orig_np.std():.6f}")
        print(f"  FreqNorm - Mean: {grad_freq_np.mean():.6f}, Std: {grad_freq_np.std():.6f}")
        print(f"  Min/Max change: {grad_orig_np.min():.6f} -> {grad_freq_np.min():.6f} / "
              f"{grad_orig_np.max():.6f} -> {grad_freq_np.max():.6f}")

        # Visualize
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original image
        axes[0, 0].imshow(img)
        axes[0, 0].set_title(f"{corruption.replace('_', ' ').title()}\nOriginal Image")
        axes[0, 0].axis('off')

        # Original gradient
        axes[0, 1].imshow(grad_orig_np.mean(axis=0), cmap='gray')
        axes[0, 1].set_title(f"Original Gradient\n(Prob: {prob_original:.3f})")
        axes[0, 1].axis('off')

        # FreqNorm gradient
        axes[0, 2].imshow(grad_freq_np.mean(axis=0), cmap='gray')
        axes[0, 2].set_title(f"FreqNorm Gradient\n(Prob: {prob_freqnorm:.3f})")
        axes[0, 2].axis('off')

        # Difference
        diff = grad_freq_np - grad_orig_np
        axes[1, 0].imshow(diff.mean(axis=0), cmap='seismic', vmin=-0.1, vmax=0.1)
        axes[1, 0].set_title(f"Difference\n(FreqNorm - Original)")
        axes[1, 0].axis('off')

        # Histogram
        axes[1, 1].hist(grad_orig_np.flatten(), bins=50, alpha=0.5, label='Original', density=True)
        axes[1, 1].hist(grad_freq_np.flatten(), bins=50, alpha=0.5, label='FreqNorm', density=True)
        axes[1, 1].set_title("Gradient Value Distribution")
        axes[1, 1].legend()
        axes[1, 1].set_xlabel("Gradient Value")
        axes[1, 1].set_ylabel("Density")

        # FFT magnitude comparison
        fft_orig = np.abs(np.fft.fftshift(np.fft.fft2(grad_orig_np.mean(axis=0))))
        fft_freq = np.abs(np.fft.fftshift(np.fft.fft2(grad_freq_np.mean(axis=0))))

        axes[1, 2].plot(fft_orig[128, :], label='Original', alpha=0.7)
        axes[1, 2].plot(fft_freq[128, :], label='FreqNorm', alpha=0.7)
        axes[1, 2].set_title("FFT Magnitude (center row)")
        axes[1, 2].set_xlabel("Frequency")
        axes[1, 2].set_ylabel("Magnitude")
        axes[1, 2].legend()
        axes[1, 2].set_yscale('log')

        plt.tight_layout()
        plt.savefig(f"/workspace/robust_deepfake_ai/debug_freqnorm_{corruption}.png", dpi=150)
        print(f"\nVisualization saved to: debug_freqnorm_{corruption}.png")
        plt.close()


def test_different_rho_values(device="cuda"):
    """Test FreqNorm with different rho values"""

    print("\n" + "="*80)
    print("Testing different rho values")
    print("="*80)

    # Original LGrad
    lgrad_original = LGrad(
        stylegan_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/karras2019stylegan-bedrooms-256x256_discriminator.pth",
        classifier_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/LGrad-Pretrained-Model/LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth",
        device=device,
        resize=256,
    )
    lgrad_original.eval()

    # Load test image (noise corrupted)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = CorruptedDataset(
        root="/workspace/robust_deepfake_ai/corrupted_dataset",
        datasets=["corrupted_test_data_progan"],
        corruptions=["gaussian_noise"],
        transform=transform,
    )

    # Get a fake sample
    sample = None
    for s in dataset.samples:
        if s["label"] == 1:  # fake
            sample = s
            break

    img = Image.open(sample["path"]).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Original prediction
    grad_orig = lgrad_original.img2grad(img_tensor)
    logits_orig = lgrad_original.classify(grad_orig)
    prob_orig = torch.sigmoid(logits_orig).item()

    print(f"\nOriginal LGrad prediction: {prob_orig:.4f}")
    print(f"\nTesting different rho values:")
    print(f"{'Rho':<10} {'Probability':<15} {'Delta':<10}")
    print("-" * 40)

    rho_values = [0.1, 0.3, 0.5, 0.7, 1.0]

    for rho in rho_values:
        freqnorm_model = create_lgrad_freqnorm(
            stylegan_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/karras2019stylegan-bedrooms-256x256_discriminator.pth",
            classifier_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/LGrad-Pretrained-Model/LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth",
            reference_stats_path="/workspace/robust_deepfake_ai/freq_reference_progan.pth",
            rho=rho,
            alpha_min=0.5,
            alpha_max=2.0,
            device=device
        )
        freqnorm_model.eval()

        logits = freqnorm_model(img_tensor)
        prob = torch.sigmoid(logits).item()
        delta = prob - prob_orig

        print(f"{rho:<10.1f} {prob:<15.4f} {delta:+.4f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Analyze FreqNorm effect
    analyze_freqnorm_effect(device)

    # Test different rho values
    test_different_rho_values(device)

    print("\n" + "="*80)
    print("Debug complete!")
    print("="*80)
