"""
1. Real vs Fake gradient 비교
2. Noise 이미지에 다양한 Huber-TV 적용 결과 비교
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
def load_sample_images():
    """Real, Fake, Noisy 이미지 로드"""
    root = Path("/workspace/robust_deepfake_ai/corrupted_dataset")
    dataset = "corrupted_test_data_deepfake"

    # Real 이미지
    real_path = root / dataset / "original" / "real"
    real_imgs = list(real_path.glob("*.png"))[:1]
    real_img = Image.open(real_imgs[0]).convert("RGB") if real_imgs else None

    # Fake 이미지
    fake_path = root / dataset / "original" / "fake"
    fake_imgs = list(fake_path.glob("*.png"))[:1]
    fake_img = Image.open(fake_imgs[0]).convert("RGB") if fake_imgs else None

    # Noisy 이미지 (Gaussian noise)
    noisy_path = root / dataset / "gaussian_noise" / "fake"
    noisy_imgs = list(noisy_path.glob("*.png"))[:1]
    noisy_img = Image.open(noisy_imgs[0]).convert("RGB") if noisy_imgs else None

    print(f"Real: {real_imgs[0] if real_imgs else 'None'}")
    print(f"Fake: {fake_imgs[0] if fake_imgs else 'None'}")
    print(f"Noisy: {noisy_imgs[0] if noisy_imgs else 'None'}")

    return real_img, fake_img, noisy_img

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 이미지 로드
print("Loading sample images...")
real_img, fake_img, noisy_img = load_sample_images()

real_tensor = transform(real_img).unsqueeze(0)
fake_tensor = transform(fake_img).unsqueeze(0)
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
    print(f"View {k}: λ={lam:.5f}, δ={delta:.5f}, iter={iters}, step={step:.2f}")

augmenter = StochasticAugmenter(config)

# Part 1: Real vs Fake gradient 추출
print("\n" + "="*60)
print("Part 1: Extracting gradients from Real and Fake")
print("="*60)

real_grad = lgrad.img2grad(real_tensor)
fake_grad = lgrad.img2grad(fake_tensor)

print(f"Real gradient range: [{real_grad.min():.3f}, {real_grad.max():.3f}]")
print(f"Fake gradient range: [{fake_grad.min():.3f}, {fake_grad.max():.3f}]")

# Part 2: Noisy 이미지 → gradient 추출 → gradient에 Huber-TV 적용
print("\n" + "="*60)
print("Part 2: Extracting gradient from noisy image, then apply Huber-TV")
print("="*60)

# Step 1: Noisy 이미지에서 gradient 추출
noisy_grad = lgrad.img2grad(noisy_tensor)
print(f"Noisy gradient range: [{noisy_grad.min():.3f}, {noisy_grad.max():.3f}]")

# Step 2: Gradient에 다양한 Huber-TV 적용
denoised_grads = []
for k, (lam, delta, iters, step) in enumerate(config.param_sets):
    if k == 0:
        denoised = noisy_grad
    else:
        denoised = augmenter.apply_anisotropic_huber_tv(
            noisy_grad,
            lambda_tv=lam,
            huber_delta=delta,
            iterations=iters,
            step_size=step
        )
    denoised_grads.append(denoised)
    print(f"View {k}: λ={lam:.5f} → gradient range [{denoised.min():.3f}, {denoised.max():.3f}]")

# 시각화
fig = plt.figure(figsize=(20, 10))

# Row 1: Real vs Fake 원본 이미지와 gradient
ax1 = plt.subplot(3, 5, 1)
ax1.imshow(real_tensor[0].permute(1, 2, 0).cpu().numpy())
ax1.set_title("Real Image", fontsize=14, fontweight='bold')
ax1.axis('off')

ax2 = plt.subplot(3, 5, 2)
ax2.imshow(real_grad[0].permute(1, 2, 0).cpu().numpy())
ax2.set_title("Real Gradient", fontsize=14, fontweight='bold')
ax2.axis('off')

ax3 = plt.subplot(3, 5, 3)
ax3.axis('off')

ax4 = plt.subplot(3, 5, 4)
ax4.imshow(fake_tensor[0].permute(1, 2, 0).cpu().numpy())
ax4.set_title("Fake Image", fontsize=14, fontweight='bold')
ax4.axis('off')

ax5 = plt.subplot(3, 5, 5)
ax5.imshow(fake_grad[0].permute(1, 2, 0).cpu().numpy())
ax5.set_title("Fake Gradient", fontsize=14, fontweight='bold')
ax5.axis('off')

# Row 2: Noisy 이미지
ax = plt.subplot(3, 5, 6)
ax.imshow(noisy_tensor[0].permute(1, 2, 0).cpu().numpy())
ax.set_title("Noisy Image\n(Gaussian Noise)", fontsize=12, fontweight='bold')
ax.axis('off')

# 나머지는 빈 공간
for k in range(1, 5):
    ax = plt.subplot(3, 5, 6 + k)
    ax.axis('off')

# Row 3: Noisy gradient에 다양한 Huber-TV 적용
for k in range(config.K):
    ax = plt.subplot(3, 5, 11 + k)
    grad_np = denoised_grads[k][0].permute(1, 2, 0).cpu().numpy()
    ax.imshow(grad_np)
    lam, delta, _, _ = config.param_sets[k]
    if k == 0:
        ax.set_title(f"Noisy Gradient\n(Original)", fontsize=12, fontweight='bold')
    else:
        ax.set_title(f"Gradient + Huber-TV\nλ={lam:.3f}, δ={delta:.3f}", fontsize=11)
    ax.axis('off')

plt.tight_layout()
output_path = "/workspace/robust_deepfake_ai/comparison_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved comparison to: {output_path}")

# 추가: 간단한 비교 (Real/Fake gradient + Noisy image Huber-TV)
fig2, axes = plt.subplots(2, 5, figsize=(20, 8))

# Row 1: Real vs Fake gradient
axes[0, 0].imshow(real_tensor[0].permute(1, 2, 0).cpu().numpy())
axes[0, 0].set_title("Real Image", fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(real_grad[0].permute(1, 2, 0).cpu().numpy())
axes[0, 1].set_title("Real Gradient", fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].axis('off')

axes[0, 3].imshow(fake_tensor[0].permute(1, 2, 0).cpu().numpy())
axes[0, 3].set_title("Fake Image", fontsize=14, fontweight='bold')
axes[0, 3].axis('off')

axes[0, 4].imshow(fake_grad[0].permute(1, 2, 0).cpu().numpy())
axes[0, 4].set_title("Fake Gradient", fontsize=14, fontweight='bold')
axes[0, 4].axis('off')

# Row 2: Noisy gradient에 Huber-TV 적용
for k in range(config.K):
    grad_np = denoised_grads[k][0].permute(1, 2, 0).cpu().numpy()
    axes[1, k].imshow(grad_np)
    lam, delta, _, _ = config.param_sets[k]
    if k == 0:
        axes[1, k].set_title(f"Noisy Gradient\n(Original)", fontsize=12, fontweight='bold')
    else:
        axes[1, k].set_title(f"Gradient + Huber-TV\nλ={lam:.3f}", fontsize=11)
    axes[1, k].axis('off')

plt.tight_layout()
output_path2 = "/workspace/robust_deepfake_ai/comparison_simple.png"
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"✓ Saved simple comparison to: {output_path2}")

print("\n" + "="*60)
print("Done!")
print("="*60)
