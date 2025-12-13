"""
LGrad Gradient Visualization Script
각 corruption 타입별로 원본 이미지와 gradient 이미지를 시각화합니다.
"""

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model.LGrad.lgrad_model import LGrad
from utils.data.dataset import CorruptedDataset


def visualize_gradients_by_corruption():
    """각 corruption 타입별로 gradient 이미지 시각화"""

    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # LGrad 모델 로드
    print("Loading LGrad model...")
    model = LGrad(
        stylegan_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/karras2019stylegan-bedrooms-256x256_discriminator.pth",
        classifier_weights="/workspace/robust_deepfake_ai/model/LGrad/weights/LGrad-Pretrained-Model/LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth",
        device=device,
        resize=256,
    )
    model.eval()
    print("Model loaded successfully!")

    # 데이터셋 로드 (256x256 resize)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    print("\nLoading dataset...")
    dataset = CorruptedDataset(
        root="/workspace/robust_deepfake_ai/corrupted_dataset",
        datasets=["corrupted_test_data_progan"],  # ProGAN 데이터만 사용 (빠른 로딩)
        transform=transform,
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # corruption 타입 리스트
    corruptions = CorruptedDataset.CORRUPTIONS
    print(f"\nCorruption types: {corruptions}")

    # corruption별로 샘플 하나씩 선택
    samples_by_corruption = {}
    for sample in dataset.samples:
        corruption = sample["corruption"]
        if corruption not in samples_by_corruption:
            samples_by_corruption[corruption] = sample
        if len(samples_by_corruption) == len(corruptions):
            break

    print(f"\nSelected {len(samples_by_corruption)} samples")

    # 시각화
    fig, axes = plt.subplots(len(corruptions), 2, figsize=(10, 3.5 * len(corruptions)))
    fig.suptitle("LGrad: Original Image vs Gradient Image (by Corruption Type)",
                 fontsize=16, fontweight='bold')

    # Note: gradient 계산을 위해 torch.no_grad()를 사용하지 않음
    for idx, corruption in enumerate(corruptions):
        if corruption not in samples_by_corruption:
            print(f"Warning: No sample found for corruption '{corruption}'")
            continue

        sample_info = samples_by_corruption[corruption]

        # 이미지 로드
        from PIL import Image
        img = Image.open(sample_info["path"]).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Gradient 추출
        print(f"Processing {corruption}...")
        logits, grad = model(img_tensor, return_grad=True)
        prob = torch.sigmoid(logits).item()

        # 이미지 변환 (tensor -> numpy)
        img_np = img_tensor.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()
        grad_np = grad.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()

        # 원본 이미지
        axes[idx, 0].imshow(img_np)
        axes[idx, 0].set_title(f"{corruption.replace('_', ' ').title()}\nOriginal Image",
                               fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')

        # Gradient 이미지
        axes[idx, 1].imshow(grad_np)
        axes[idx, 1].set_title(f"Gradient Image\n(Fake prob: {prob:.3f})",
                               fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')

        # 메타데이터 표시
        label = "Fake" if sample_info["label"] == 1 else "Real"
        dataset_name = sample_info["dataset"]
        axes[idx, 0].text(0.02, 0.98, f"{dataset_name} | {label}",
                        transform=axes[idx, 0].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # 저장
    output_path = "/workspace/robust_deepfake_ai/gradient_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    plt.show()
    print("\nDone!")


if __name__ == "__main__":
    visualize_gradients_by_corruption()
