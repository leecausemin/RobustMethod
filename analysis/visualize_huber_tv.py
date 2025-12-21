"""
Anisotropic Huber-TV Denoising 시각화
"""
import sys
sys.path.insert(0, '/workspace/robust_deepfake_ai')

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pathlib import Path

from model.method.sgs import SGSConfig, StochasticAugmenter

# 샘플 이미지 로드
def load_sample_image(corruption="gaussian_noise"):
    """코럽션이 적용된 샘플 이미지 로드"""
    root = Path("/workspace/robust_deepfake_ai/corrupted_dataset")

    # 첫 번째 데이터셋에서 샘플 찾기
    datasets = ["corrupted_test_data_deepfake", "corrupted_test_data_progan", "corrupted_test_data_stylegan"]

    for dataset in datasets:
        img_path = root / dataset / corruption / "fake"
        if img_path.exists():
            imgs = list(img_path.glob("*.png"))[:1]
            if imgs:
                print(f"Loading image: {imgs[0]}")
                return Image.open(imgs[0]).convert("RGB")

    print("Warning: No sample found, creating random image")
    return Image.new("RGB", (256, 256), color=(128, 128, 128))

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 이미지 로드
print("Loading sample images...")
original_img = load_sample_image("original")
noisy_img = load_sample_image("gaussian_noise")

original_tensor = transform(original_img).unsqueeze(0)  # [1, 3, 256, 256]
noisy_tensor = transform(noisy_img).unsqueeze(0)

print(f"Image shape: {noisy_tensor.shape}")

# SGS Config 생성
config = SGSConfig(
    K=5,
    huber_tv_lambda=0.05,
    huber_tv_delta=0.01,
    huber_tv_iterations=5,
    huber_tv_step_size=0.2,
    device="cpu"
)

print(f"\nGenerated {config.K} parameter sets:")
for k, (lam, delta, iters, step) in enumerate(config.param_sets):
    print(f"View {k}: λ={lam:.5f}, δ={delta:.5f}, iter={iters}, step={step:.2f}")

# Augmenter 생성
augmenter = StochasticAugmenter(config)

# 각 파라미터로 denoising 적용
denoised_images = []
for k, (lam, delta, iters, step) in enumerate(config.param_sets):
    print(f"\nProcessing view {k}...")
    if k == 0:
        # 원본
        denoised = noisy_tensor
    else:
        # Huber-TV 적용
        denoised = augmenter.apply_anisotropic_huber_tv(
            noisy_tensor,
            lambda_tv=lam,
            huber_delta=delta,
            iterations=iters,
            step_size=step
        )
    denoised_images.append(denoised)

# 시각화
fig, axes = plt.subplots(2, config.K, figsize=(config.K * 4, 8))

# 첫 번째 행: Original image로 테스트
print("\n" + "="*60)
print("Row 1: Original (no corruption)")
print("="*60)
for k in range(config.K):
    lam, delta, iters, step = config.param_sets[k]

    if k == 0:
        result = original_tensor
    else:
        result = augmenter.apply_anisotropic_huber_tv(
            original_tensor,
            lambda_tv=lam,
            huber_delta=delta,
            iterations=iters,
            step_size=step
        )

    img_np = result[0].permute(1, 2, 0).cpu().numpy()
    axes[0, k].imshow(img_np)
    axes[0, k].set_title(f"View {k}\nλ={lam:.3f}, δ={delta:.3f}")
    axes[0, k].axis('off')

# 두 번째 행: Noisy image로 테스트
print("\n" + "="*60)
print("Row 2: Gaussian Noise Corrupted")
print("="*60)
for k in range(config.K):
    img_np = denoised_images[k][0].permute(1, 2, 0).cpu().numpy()
    axes[1, k].imshow(img_np)

    lam, delta, iters, step = config.param_sets[k]
    axes[1, k].set_title(f"View {k}\nλ={lam:.3f}, δ={delta:.3f}")
    axes[1, k].axis('off')

plt.tight_layout()
output_path = "/workspace/robust_deepfake_ai/huber_tv_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization to: {output_path}")

# 추가: 평균 이미지 계산
averaged = torch.stack(denoised_images, dim=0).mean(dim=0)

fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))

# 원본
axes2[0].imshow(noisy_tensor[0].permute(1, 2, 0).cpu().numpy())
axes2[0].set_title("Original (with Gaussian Noise)")
axes2[0].axis('off')

# View 1 (가장 약한 denoising)
axes2[1].imshow(denoised_images[1][0].permute(1, 2, 0).cpu().numpy())
lam, delta, _, _ = config.param_sets[1]
axes2[1].set_title(f"Single View (λ={lam:.3f})")
axes2[1].axis('off')

# 평균
axes2[2].imshow(averaged[0].permute(1, 2, 0).cpu().numpy())
axes2[2].set_title(f"Averaged ({config.K} views)")
axes2[2].axis('off')

plt.tight_layout()
output_path2 = "/workspace/robust_deepfake_ai/huber_tv_averaged.png"
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"✓ Saved averaged comparison to: {output_path2}")

print("\n" + "="*60)
print("Done! Check the saved images.")
print("="*60)
