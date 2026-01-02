"""
Visualize SAS (Structure-Aware Sharpening) effects on artifacts

각 corruption 타입별로 SAS 적용 전/후 artifact 비교 시각화
"""

import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

from model.LGrad.lgrad_model import LGrad
from model.method.sas import UnifiedSAS, SASConfig


def load_image(image_path, device="cuda"):
    """이미지 로드 및 전처리"""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


def tensor_to_numpy(tensor):
    """Tensor를 numpy 이미지로 변환 (시각화용)"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img


def visualize_sas_comparison(
    lgrad_model,
    sas_model,
    image_paths,
    corruption_types,
    save_path="sas_comparison.png"
):
    """
    SAS 적용 전/후 비교 시각화

    Args:
        lgrad_model: Original LGrad model
        sas_model: SAS-wrapped model
        image_paths: dict {corruption_type: image_path}
        corruption_types: list of corruption types to visualize
        save_path: 저장 경로
    """
    n_corruptions = len(corruption_types)

    # Figure 생성
    # Columns: [Original Image, Original Artifact, Enhanced Artifact (SAS)]
    fig, axes = plt.subplots(n_corruptions, 3, figsize=(12, 4 * n_corruptions))

    if n_corruptions == 1:
        axes = axes.reshape(1, -1)

    for i, corruption_type in enumerate(corruption_types):
        print(f"Processing {corruption_type}...")

        # 이미지 로드
        img_path = image_paths[corruption_type]
        img = load_image(img_path, device=sas_model.device)

        # 1. Original artifact (LGrad)
        artifact_original = lgrad_model.img2grad(img)  # [1, 3, 256, 256]

        # 2. SAS enhanced artifact (Robust Fusion only)
        with torch.no_grad():
            _, artifact_enhanced, masks = sas_model(
                img,
                return_artifact=True,
                return_masks=True
            )

        # Convert to numpy
        img_np = tensor_to_numpy(img)
        artifact_orig_np = tensor_to_numpy(artifact_original)
        artifact_enh_np = tensor_to_numpy(artifact_enhanced)

        # Get fused result (same as enhanced now)
        artifact_fused_np = None
        if 'artifact_fused' in masks:
            artifact_fused_np = tensor_to_numpy(masks['artifact_fused'])

        # Plot
        # Column 0: Original Image
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"{corruption_type}\nOriginal Image", fontsize=12)
        axes[i, 0].axis('off')

        # Column 1: Original Artifact
        axes[i, 1].imshow(artifact_orig_np)
        axes[i, 1].set_title("Original Artifact\n(LGrad - No SAS)", fontsize=12)
        axes[i, 1].axis('off')

        # Column 2: Enhanced Artifact (SAS with Robust Fusion)
        axes[i, 2].imshow(artifact_enh_np)
        axes[i, 2].set_title("Enhanced Artifact\n(SAS - Robust Fusion)", fontsize=12)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to {save_path}")
    plt.close()


def main():
    """Main visualization script"""

    # ===== 설정 =====
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 경로
    stylegan_weights = "model/LGrad/weights/karras2019stylegan-bedrooms-256x256_discriminator.pth"
    classifier_weights = "model/LGrad/weights/LGrad-Pretrained-Model/LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth"

    # ProGAN fake 이미지 경로 설정
    gan_type = "progan"  # 또는 "stylegan", "stylegan2" 등
    base_dir = Path(f"corrupted_dataset/corrupted_test_data_{gan_type}")

    # Corruption 타입 (존재하는 것만)
    corruption_types = [
        "original",
        "contrast",
        "gaussian_noise",
        "motion_blur",
        "pixelate",
        "jpeg_compression",
        "fog"
    ]

    # 첫 번째 fake 이미지 찾기
    image_paths = {}
    for corruption_type in corruption_types:
        corruption_dir = base_dir / corruption_type / "fake"
        if corruption_dir.exists():
            # 첫 번째 이미지 사용
            images = sorted(list(corruption_dir.glob("*.png")))
            if images:
                image_paths[corruption_type] = str(images[0])
                print(f"Found {corruption_type}: {images[0].name}")
        else:
            print(f"Warning: {corruption_dir} does not exist, skipping...")

    if not image_paths:
        print("Error: No images found!")
        print(f"Please check if {base_dir} exists and contains corrupted images.")
        return

    # 실제 존재하는 corruption만 사용
    corruption_types = [ct for ct in corruption_types if ct in image_paths]

    print(f"\n=== Visualizing {len(corruption_types)} corruption types ===\n")

    # ===== 모델 로드 =====
    print("Loading LGrad model...")
    lgrad = LGrad(
        stylegan_weights=stylegan_weights,
        classifier_weights=classifier_weights,
        device=device,
        resize=256,
    )
    lgrad.eval()

    # ===== View 다양성 증가 테스트 (input에 적용) =====
    configs_to_test = [
        {
            "name": "diverse_input",
            "config": SASConfig(
                K=7, model="LGrad", denoise_target="input",  # input에 적용
                lambda_range=(0.3, 2.0), jitter_strength=0.2,
                consistency_metric="mad", mask_type="soft", gamma=20.0,
                coherence_method="structure_tensor", eta=1.5, gaussian_sigma=1.0,
                huber_tv_lambda=0.05, huber_tv_delta=0.01, device=device,
            )
        },
        {
            "name": "very_diverse_input",
            "config": SASConfig(
                K=7, model="LGrad", denoise_target="input",
                lambda_range=(0.1, 3.0), jitter_strength=0.3,  # 매우 넓은 범위
                consistency_metric="mad", mask_type="soft", gamma=20.0,
                coherence_method="structure_tensor", eta=1.5, gaussian_sigma=1.0,
                huber_tv_lambda=0.05, huber_tv_delta=0.01, device=device,
            )
        },
        {
            "name": "diverse_input_hard",
            "config": SASConfig(
                K=7, model="LGrad", denoise_target="input",
                lambda_range=(0.3, 2.0), jitter_strength=0.2,
                consistency_metric="mad", mask_type="hard", tau=0.1,
                coherence_method="structure_tensor", eta=1.0, gaussian_sigma=1.0,
                huber_tv_lambda=0.05, huber_tv_delta=0.01, device=device,
            )
        },
        {
            "name": "diverse_input_weak_coh",
            "config": SASConfig(
                K=7, model="LGrad", denoise_target="input",
                lambda_range=(0.3, 2.0), jitter_strength=0.2,
                consistency_metric="mad", mask_type="soft", gamma=20.0,
                coherence_method="structure_tensor", eta=0.5, gaussian_sigma=1.5,
                huber_tv_lambda=0.05, huber_tv_delta=0.01, device=device,
            )
        },
    ]

    for test_config in configs_to_test:
        print(f"\n{'='*60}")
        print(f"Testing configuration: {test_config['name']}")
        print(f"{'='*60}")

        print("Creating SAS model...")
        sas = UnifiedSAS(lgrad, test_config['config'])
        sas.eval()

        print("\nGenerating visualization...")
        save_path = f"sas_comparison_{gan_type}_{test_config['name']}.png"
        visualize_sas_comparison(
            lgrad_model=lgrad,
            sas_model=sas,
            image_paths=image_paths,
            corruption_types=corruption_types,
            save_path=save_path
        )

    print("\n" + "="*60)
    print("✓ All configurations tested!")
    print("="*60)


if __name__ == "__main__":
    main()
