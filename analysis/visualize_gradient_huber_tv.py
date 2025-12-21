"""
Gradient에 Anisotropic Huber-TV Denoising 적용 시각화
"""
import sys
sys.path.insert(0, '/workspace/robust_deepfake_ai')

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pathlib import Path

from model.method.sgs import SGSConfig, StochasticAugmenter
from model.LGrad.lgrad_model import LGrad

# 샘플 이미지 로드
def load_sample_image(corruption="gaussian_noise"):
    """코럽션이 적용된 샘플 이미지 로드"""
    root = Path("/workspace/robust_deepfake_ai/corrupted_dataset")
    datasets = ["corrupted_test_data_deepfake", "corrupted_test_data_progan"]

    for dataset in datasets:
        img_path = root / dataset / corruption / "fake"
        if img_path.exists():
            imgs = list(img_path.glob("*.png"))[:1]
            if imgs:
                print(f"Loading image: {imgs[0]}")
                return Image.open(imgs[0]).convert("RGB")

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

# LGrad 모델 로드
print("\nLoading LGrad model...")
STYLEGAN_WEIGHTS = "/workspace/robust_deepfake_ai/model/LGrad/weights/karras2019stylegan-bedrooms-256x256_discriminator.pth"
CLASSIFIER_WEIGHTS = "/workspace/robust_deepfake_ai/model/LGrad/weights/LGrad-Pretrained-Model/LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth"

lgrad = LGrad(
    stylegan_weights=STYLEGAN_WEIGHTS,
    classifier_weights=CLASSIFIER_WEIGHTS,
    device="cpu"
)
lgrad.eval()

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

# Gradient 추출 함수
def extract_and_denoise_gradients(img_tensor, lgrad_model, augmenter, param_sets):
    """이미지 → gradient 추출 → K개의 Huber-TV denoising"""
    # Step 1: 원본 이미지에서 gradient 추출
    grad = lgrad_model.img2grad(img_tensor)  # [1, 3, 256, 256]

    print(f"Gradient shape: {grad.shape}, range: [{grad.min():.3f}, {grad.max():.3f}]")

    # Step 2: K개의 다른 파라미터로 gradient denoising
    denoised_grads = []
    for k, (lam, delta, iters, step) in enumerate(param_sets):
        print(f"Denoising gradient with view {k} params...")
        if k == 0:
            grad_k = grad
        else:
            grad_k = augmenter.apply_anisotropic_huber_tv(
                grad,
                lambda_tv=lam,
                huber_delta=delta,
                iterations=iters,
                step_size=step
            )
        denoised_grads.append(grad_k)

    return grad, denoised_grads

# Original image로 테스트
print("\n" + "="*60)
print("Processing Original Image (no corruption)")
print("="*60)
orig_grad, orig_denoised = extract_and_denoise_gradients(
    original_tensor, lgrad, augmenter, config.param_sets
)

# Noisy image로 테스트
print("\n" + "="*60)
print("Processing Noisy Image (Gaussian noise)")
print("="*60)
noisy_grad, noisy_denoised = extract_and_denoise_gradients(
    noisy_tensor, lgrad, augmenter, config.param_sets
)

# 시각화
fig, axes = plt.subplots(3, config.K, figsize=(config.K * 4, 12))

# Row 0: 원본 이미지들
axes[0, 0].imshow(original_tensor[0].permute(1, 2, 0).cpu().numpy())
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

for k in range(1, config.K):
    axes[0, k].axis('off')

# Row 1: Original gradient에 K개의 denoising
for k in range(config.K):
    grad_np = orig_denoised[k][0].permute(1, 2, 0).cpu().numpy()
    axes[1, k].imshow(grad_np)
    lam, delta, _, _ = config.param_sets[k]
    axes[1, k].set_title(f"View {k} Gradient\nλ={lam:.3f}, δ={delta:.3f}")
    axes[1, k].axis('off')

# Row 2: Noisy gradient에 K개의 denoising
axes[2, 0].text(0.5, 1.1, "Gaussian Noise →", transform=axes[2, 0].transAxes,
                ha='center', va='bottom', fontsize=12, fontweight='bold')
for k in range(config.K):
    grad_np = noisy_denoised[k][0].permute(1, 2, 0).cpu().numpy()
    axes[2, k].imshow(grad_np)
    lam, delta, _, _ = config.param_sets[k]
    axes[2, k].set_title(f"View {k} Gradient\nλ={lam:.3f}, δ={delta:.3f}")
    axes[2, k].axis('off')

plt.tight_layout()
output_path = "/workspace/robust_deepfake_ai/gradient_huber_tv_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved gradient visualization to: {output_path}")

# 추가: Averaging 효과 비교
averaged_grad = torch.stack(noisy_denoised, dim=0).mean(dim=0)

fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))

# 원본 이미지
axes2[0].imshow(noisy_tensor[0].permute(1, 2, 0).cpu().numpy())
axes2[0].set_title("Input\n(Gaussian Noise)")
axes2[0].axis('off')

# 원본 gradient
axes2[1].imshow(noisy_grad[0].permute(1, 2, 0).cpu().numpy())
axes2[1].set_title("Original Gradient\n(noisy)")
axes2[1].axis('off')

# Single denoised gradient
axes2[2].imshow(noisy_denoised[1][0].permute(1, 2, 0).cpu().numpy())
lam, delta, _, _ = config.param_sets[1]
axes2[2].set_title(f"Single Denoised\n(λ={lam:.3f})")
axes2[2].axis('off')

# Averaged gradient
axes2[3].imshow(averaged_grad[0].permute(1, 2, 0).cpu().numpy())
axes2[3].set_title(f"Averaged\n({config.K} views)")
axes2[3].axis('off')

plt.tight_layout()
output_path2 = "/workspace/robust_deepfake_ai/gradient_averaged_comparison.png"
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"✓ Saved averaging comparison to: {output_path2}")

print("\n" + "="*60)
print("Done! Check the saved images.")
print("="*60)
