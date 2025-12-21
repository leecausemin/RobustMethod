"""
Image-space vs Gradient-space Huber-TV 비교
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
def load_noisy_image():
    root = Path("/workspace/robust_deepfake_ai/corrupted_dataset")
    dataset = "corrupted_test_data_deepfake"

    noisy_path = root / dataset / "gaussian_noise" / "fake"
    noisy_imgs = list(noisy_path.glob("*.png"))[:1]
    noisy_img = Image.open(noisy_imgs[0]).convert("RGB") if noisy_imgs else None

    print(f"Noisy: {noisy_imgs[0]}")
    return noisy_img

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 이미지 로드
print("Loading noisy image...")
noisy_img = load_noisy_image()
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

# SGS Config
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
    print(f"View {k}: λ={lam:.5f}, δ={delta:.5f}")

augmenter = StochasticAugmenter(config)

# Approach 1: Image-space denoising (이미지에 Huber-TV → gradient)
print("\n" + "="*60)
print("Approach 1: Image-space denoising (Image → Huber-TV → Gradient)")
print("="*60)

image_denoised_grads = []
for k, (lam, delta, iters, step) in enumerate(config.param_sets):
    if k == 0:
        # 원본 이미지
        img_denoised = noisy_tensor
    else:
        # 이미지에 Huber-TV 적용
        img_denoised = augmenter.apply_anisotropic_huber_tv(
            noisy_tensor,
            lambda_tv=lam,
            huber_delta=delta,
            iterations=iters,
            step_size=step
        )

    # Denoised 이미지에서 gradient 추출
    grad = lgrad.img2grad(img_denoised)
    image_denoised_grads.append((img_denoised, grad))
    print(f"View {k}: Image [{img_denoised.min():.3f}, {img_denoised.max():.3f}] → Gradient [{grad.min():.3f}, {grad.max():.3f}]")

# Approach 2: Gradient-space denoising (gradient → Huber-TV)
print("\n" + "="*60)
print("Approach 2: Gradient-space denoising (Image → Gradient → Huber-TV)")
print("="*60)

# 원본 gradient 추출
original_grad = lgrad.img2grad(noisy_tensor)
print(f"Original gradient: [{original_grad.min():.3f}, {original_grad.max():.3f}]")

gradient_denoised_grads = []
for k, (lam, delta, iters, step) in enumerate(config.param_sets):
    if k == 0:
        grad_denoised = original_grad
    else:
        # Gradient에 Huber-TV 적용
        grad_denoised = augmenter.apply_anisotropic_huber_tv(
            original_grad,
            lambda_tv=lam,
            huber_delta=delta,
            iterations=iters,
            step_size=step
        )

    gradient_denoised_grads.append(grad_denoised)
    print(f"View {k}: Gradient after Huber-TV [{grad_denoised.min():.3f}, {grad_denoised.max():.3f}]")

# 시각화
fig, axes = plt.subplots(3, config.K, figsize=(config.K * 4, 12))

# Row 0: 원본 noisy 이미지와 gradient
axes[0, 0].imshow(noisy_tensor[0].permute(1, 2, 0).cpu().numpy())
axes[0, 0].set_title("Noisy Image\n(Original)", fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(original_grad[0].permute(1, 2, 0).cpu().numpy())
axes[0, 1].set_title("Noisy Gradient\n(Original)", fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

for k in range(2, config.K):
    axes[0, k].axis('off')

# Row 1: Approach 1 (Image → Huber-TV → Gradient)
for k in range(config.K):
    img_denoised, grad = image_denoised_grads[k]

    # Gradient 시각화
    axes[1, k].imshow(grad[0].permute(1, 2, 0).cpu().numpy())
    lam, delta, _, _ = config.param_sets[k]
    if k == 0:
        axes[1, k].set_title(f"[Approach 1]\nOriginal", fontsize=11, fontweight='bold')
    else:
        axes[1, k].set_title(f"[Approach 1]\nImage Huber-TV (λ={lam:.3f})\n→ Gradient", fontsize=10)
    axes[1, k].axis('off')

# Row 2: Approach 2 (Image → Gradient → Huber-TV)
for k in range(config.K):
    grad_denoised = gradient_denoised_grads[k]

    axes[2, k].imshow(grad_denoised[0].permute(1, 2, 0).cpu().numpy())
    lam, delta, _, _ = config.param_sets[k]
    if k == 0:
        axes[2, k].set_title(f"[Approach 2]\nOriginal", fontsize=11, fontweight='bold')
    else:
        axes[2, k].set_title(f"[Approach 2]\nGradient Huber-TV (λ={lam:.3f})", fontsize=10)
    axes[2, k].axis('off')

plt.tight_layout()
output_path = "/workspace/robust_deepfake_ai/approach_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved comparison to: {output_path}")

# 추가: 더 자세한 비교 (한 가지 λ 값으로)
selected_k = 2  # λ=0.04 정도
lam, delta, _, _ = config.param_sets[selected_k]

fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8))

# Approach 1
img_den, grad_from_img = image_denoised_grads[selected_k]

axes2[0, 0].imshow(noisy_tensor[0].permute(1, 2, 0).cpu().numpy())
axes2[0, 0].set_title("Noisy Image", fontsize=12)
axes2[0, 0].axis('off')

axes2[0, 1].imshow(img_den[0].permute(1, 2, 0).cpu().numpy())
axes2[0, 1].set_title(f"Image after Huber-TV\n(λ={lam:.3f})", fontsize=12)
axes2[0, 1].axis('off')

axes2[0, 2].imshow(grad_from_img[0].permute(1, 2, 0).cpu().numpy())
axes2[0, 2].set_title("[Approach 1] Gradient", fontsize=12, fontweight='bold')
axes2[0, 2].axis('off')

# Approach 2
grad_den = gradient_denoised_grads[selected_k]

axes2[1, 0].imshow(noisy_tensor[0].permute(1, 2, 0).cpu().numpy())
axes2[1, 0].set_title("Noisy Image", fontsize=12)
axes2[1, 0].axis('off')

axes2[1, 1].imshow(original_grad[0].permute(1, 2, 0).cpu().numpy())
axes2[1, 1].set_title("Original Gradient", fontsize=12)
axes2[1, 1].axis('off')

axes2[1, 2].imshow(grad_den[0].permute(1, 2, 0).cpu().numpy())
axes2[1, 2].set_title(f"[Approach 2] Gradient\nafter Huber-TV (λ={lam:.3f})", fontsize=12, fontweight='bold')
axes2[1, 2].axis('off')

fig2.text(0.02, 0.75, "Approach 1:\nImage Huber-TV → Grad", fontsize=14, fontweight='bold', va='center')
fig2.text(0.02, 0.25, "Approach 2:\nGrad → Huber-TV", fontsize=14, fontweight='bold', va='center')

plt.tight_layout(rect=[0.05, 0, 1, 1])
output_path2 = "/workspace/robust_deepfake_ai/approach_detailed.png"
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"✓ Saved detailed comparison to: {output_path2}")

print("\n" + "="*60)
print("Done!")
print("="*60)
