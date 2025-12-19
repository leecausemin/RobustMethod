"""
Debug: Gaussian Noise에서 FreqNorm이 왜 실패하는지 분석
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from model.LGrad.lgrad_model import LGrad
from model.method.freq_norm import create_lgrad_freqnorm, FREQUENCY_BANDS
from utils.data.dataset import CorruptedDataset


def analyze_gaussian_noise_failure(device="cuda"):
    """Gaussian Noise에서 FreqNorm 실패 원인 분석"""

    print("Loading models...")

    # Original LGrad
    lgrad = LGrad(
        stylegan_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/karras2019stylegan-bedrooms-256x256_discriminator.pth",
        classifier_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/LGrad-Pretrained-Model/LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth",
        device=device,
        resize=256,
    )
    lgrad.eval()

    # FreqNorm
    freqnorm = create_lgrad_freqnorm(
        stylegan_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/karras2019stylegan-bedrooms-256x256_discriminator.pth",
        classifier_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/LGrad-Pretrained-Model/LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth",
        reference_stats_path="/workspace/robust_deepfake_ai/freq_reference_progan.pth",
        rho=0.3,
        device=device
    )
    freqnorm.eval()

    print("Models loaded!\n")

    # Load reference stats
    ref_stats = torch.load("/workspace/robust_deepfake_ai/freq_reference_progan.pth", weights_only=False)
    E_ref = ref_stats['mean']

    print("Reference Energy (ProGAN clean):")
    for i, (r_min, r_max, name) in enumerate(FREQUENCY_BANDS):
        print(f"  {name:12s}: {E_ref[i]:.2e}")

    # Load gaussian noise samples
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = CorruptedDataset(
        root="/workspace/robust_deepfake_ai/corrupted_dataset",
        datasets=["corrupted_test_data_progan", "corrupted_test_data_biggan"],
        corruptions=["gaussian_noise"],
        transform=transform,
    )

    # Test on multiple samples (both real and fake)
    n_samples_per_label = 5
    results = {
        'original': {'real': [], 'fake': []},
        'freqnorm': {'real': [], 'fake': []},
    }

    print(f"\n{'='*80}")
    print(f"Testing on {n_samples_per_label*2} samples ({n_samples_per_label} real, {n_samples_per_label} fake)")
    print(f"{'='*80}")

    # Collect samples by label
    real_samples = [s for s in dataset.samples if s['label'] == 0][:n_samples_per_label]
    fake_samples = [s for s in dataset.samples if s['label'] == 1][:n_samples_per_label]
    all_samples = real_samples + fake_samples

    for i, sample in enumerate(all_samples):
        img = Image.open(sample["path"]).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        label = sample["label"]
        label_str = "Fake" if label == 1 else "Real"

        # Original LGrad
        grad_orig = lgrad.img2grad(img_tensor)
        with torch.no_grad():
            logits_orig = lgrad.classify(grad_orig)
            prob_orig = torch.sigmoid(logits_orig).item()

        # FreqNorm
        grad_freq = freqnorm.img2grad_with_freqnorm(img_tensor)
        with torch.no_grad():
            logits_freq = freqnorm.lgrad.classify(grad_freq)
            prob_freq = torch.sigmoid(logits_freq).item()

        # Compute band energies
        grad_orig_fft = torch.fft.fftshift(torch.fft.fft2(grad_orig))
        grad_freq_fft = torch.fft.fftshift(torch.fft.fft2(grad_freq))

        print(f"\nSample {i+1} ({label_str}):")
        print(f"  Original: {prob_orig:.4f} | FreqNorm: {prob_freq:.4f} | Diff: {prob_freq-prob_orig:+.4f}")

        # Compute energies for each band
        print(f"  {'Band':<12} {'Original':>12} {'FreqNorm':>12} {'Reference':>12} {'Gain':>8}")
        print(f"  {'-'*60}")

        from model.method.freq_norm import create_radial_frequency_mask

        for band_idx, (r_min, r_max, band_name) in enumerate(FREQUENCY_BANDS):
            mask = create_radial_frequency_mask((256, 256), r_min, r_max, device=device)

            if mask.sum() > 0:
                # Original energy
                power_orig = torch.abs(grad_orig_fft) ** 2
                E_orig = (power_orig[:, :, mask].mean()).item()

                # FreqNorm energy
                power_freq = torch.abs(grad_freq_fft) ** 2
                E_freq = (power_freq[:, :, mask].mean()).item()

                # Expected gain
                gain_expected = (E_ref[band_idx] / (E_orig + 1e-8)) ** 0.3
                gain_expected = torch.clamp(gain_expected, 0.85, 1.15).item()

                # Actual change
                actual_ratio = E_freq / (E_orig + 1e-8)

                print(f"  {band_name:<12} {E_orig:>12.2e} {E_freq:>12.2e} {E_ref[band_idx]:>12.2e} {actual_ratio:>8.3f}")

        # Store results
        key = 'fake' if label == 1 else 'real'
        results['original'][key].append(prob_orig)
        results['freqnorm'][key].append(prob_freq)

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")

    for method in ['original', 'freqnorm']:
        print(f"\n{method.upper()}:")
        for label_key in ['real', 'fake']:
            probs = results[method][label_key]
            if probs:
                avg = np.mean(probs)
                print(f"  {label_key.capitalize()}: {avg:.4f} (n={len(probs)})")

    print("\n" + "="*80)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    analyze_gaussian_noise_failure(device)
