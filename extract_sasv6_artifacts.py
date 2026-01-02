"""
Extract artifacts using SASv6 preprocessor for ProGAN images
with different corruptions (original, gaussian, jpeg)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import imagecorruptions

from model.method.sasv6 import SASv6Config, SASv6Preprocessor

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_root = Path("/workspace/robust_deepfake_ai/dataset/test/progan/car")

# Create SASv6 preprocessor
config = SASv6Config(
    model="LGrad",
    gaussian_sigma=0.8,
    dct_threshold=2.5,
    dwt_threshold=0.05,
    wavelet_type='haar',
    input_mode='concat',
    residual_type='R_d',
    device=device
)

preprocessor = SASv6Preprocessor(config)

print(f"[SASv6 Artifact Extraction]")
print(f"Device: {device}")
print(f"Config: gaussian_σ={config.gaussian_sigma}, dct_thresh={config.dct_threshold}, dwt_λ={config.dwt_threshold}")
print()


def load_image(image_path, resize=256):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((resize, resize), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return img_tensor


def apply_corruption(img_np, corruption_type, severity=3):
    """
    Apply corruption to image

    Args:
        img_np: numpy array (H, W, 3) in [0, 255]
        corruption_type: 'gaussian_noise' or 'jpeg_compression'
        severity: 1-5

    Returns:
        corrupted image in [0, 255]
    """
    img_uint8 = img_np.astype(np.uint8)
    corrupted = imagecorruptions.corrupt(img_uint8, corruption_name=corruption_type, severity=severity)
    return corrupted


def extract_all_artifacts(img_tensor):
    """
    Extract all intermediate artifacts from SASv6 pipeline

    Args:
        img_tensor: (1, 3, H, W) in [0, 1]

    Returns:
        dict with all artifacts
    """
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        # Step 1: Gaussian blur
        I_g = preprocessor.gaussian_blur(img_tensor)

        # Step 2: DCT normalization
        I_d = preprocessor.block_dct_normalize(I_g)

        # Step 3: Residuals
        R_g = img_tensor - I_g
        R_d = I_g - I_d

        # Step 4: DWT refinement
        I_d_wave, R_d_wave = preprocessor.make_wavelet_refined(I_d, R_d)

    return {
        'original': img_tensor.cpu(),
        'I_g': I_g.cpu(),
        'I_d': I_d.cpu(),
        'R_g': R_g.cpu(),
        'R_d': R_d.cpu(),
        'I_d_wave': I_d_wave.cpu(),
        'R_d_wave': R_d_wave.cpu(),
    }


def normalize_for_display(tensor, mode='residual'):
    """
    Normalize tensor for visualization

    Args:
        tensor: (1, 3, H, W) or (1, C, H, W)
        mode: 'image' or 'residual'
    """
    if mode == 'image':
        # Clamp to [0, 1]
        return torch.clamp(tensor, 0, 1)
    else:  # residual
        # Normalize to [-1, 1] then shift to [0, 1]
        t_min = tensor.min()
        t_max = tensor.max()
        if t_max - t_min > 1e-6:
            normalized = (tensor - t_min) / (t_max - t_min + 1e-8)
        else:
            normalized = torch.zeros_like(tensor)
        return normalized


def visualize_artifacts(artifacts_dict, title_prefix=""):
    """
    Visualize all artifacts

    Args:
        artifacts_dict: dict with artifact tensors
        title_prefix: prefix for plot title
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"{title_prefix} - SASv6 Pipeline Artifacts", fontsize=16, fontweight='bold')

    # Row 1: Image progression
    items_row1 = [
        ('original', 'Original I', 'image'),
        ('I_g', 'Gaussian I_g', 'image'),
        ('I_d', 'DCT I_d', 'image'),
        ('I_d_wave', 'DWT I_d_wave', 'image'),
    ]

    for idx, (key, title, mode) in enumerate(items_row1):
        ax = axes[0, idx]
        tensor = artifacts_dict[key]
        img = normalize_for_display(tensor, mode)
        img_np = img[0].permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    # Row 2: Residuals
    items_row2 = [
        ('R_g', 'R_g = I - I_g\n(Gaussian residual)', 'residual'),
        ('R_d', 'R_d = I_g - I_d\n(DCT residual)', 'residual'),
        ('R_d_wave', 'R_d_wave\n(DWT refined)', 'residual'),
        ('original', 'Original (ref)', 'image'),
    ]

    for idx, (key, title, mode) in enumerate(items_row2):
        ax = axes[1, idx]
        tensor = artifacts_dict[key]
        img = normalize_for_display(tensor, mode)
        img_np = img[0].permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    return fig


# Main execution
if __name__ == "__main__":
    # Load sample images
    real_images = sorted(list((dataset_root / "0_real").glob("*.png")))[:2]
    fake_images = sorted(list((dataset_root / "1_fake").glob("*.png")))[:2]

    print(f"Real images: {len(real_images)}")
    print(f"Fake images: {len(fake_images)}")
    print()

    # Process one real and one fake image
    for label, img_path in [("REAL", real_images[0]), ("FAKE", fake_images[0])]:
        print(f"\n{'='*60}")
        print(f"Processing {label}: {img_path.name}")
        print('='*60)

        # Load original image
        img_tensor = load_image(img_path).to(device)
        img_np = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Extract artifacts for different corruptions
        corruption_configs = [
            ("Original", None, None),
            ("Gaussian Noise (σ=3)", "gaussian_noise", 3),
            ("JPEG Compression (Q=3)", "jpeg_compression", 3),
        ]

        for corruption_name, corruption_type, severity in corruption_configs:
            print(f"\n[{corruption_name}]")

            if corruption_type is None:
                # Original (no corruption)
                current_tensor = img_tensor
            else:
                # Apply corruption
                corrupted_np = apply_corruption(img_np, corruption_type, severity)
                current_tensor = torch.from_numpy(corrupted_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # Extract artifacts
            artifacts = extract_all_artifacts(current_tensor)

            # Print statistics
            print(f"  Original range: [{artifacts['original'].min():.3f}, {artifacts['original'].max():.3f}]")
            print(f"  I_g range: [{artifacts['I_g'].min():.3f}, {artifacts['I_g'].max():.3f}]")
            print(f"  I_d range: [{artifacts['I_d'].min():.3f}, {artifacts['I_d'].max():.3f}]")
            print(f"  R_g range: [{artifacts['R_g'].min():.3f}, {artifacts['R_g'].max():.3f}]")
            print(f"  R_d range: [{artifacts['R_d'].min():.3f}, {artifacts['R_d'].max():.3f}]")
            print(f"  I_d_wave range: [{artifacts['I_d_wave'].min():.3f}, {artifacts['I_d_wave'].max():.3f}]")
            print(f"  R_d_wave range: [{artifacts['R_d_wave'].min():.3f}, {artifacts['R_d_wave'].max():.3f}]")

            # Visualize
            title = f"{label} - {corruption_name}"
            fig = visualize_artifacts(artifacts, title)

            # Save figure
            save_path = f"/workspace/robust_deepfake_ai/sasv6_artifacts_{label.lower()}_{corruption_type or 'original'}.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved visualization: {save_path}")
            plt.close(fig)

    print(f"\n{'='*60}")
    print("✓ Artifact extraction complete!")
    print('='*60)
