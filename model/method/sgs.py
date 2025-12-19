"""
Stochastic Gradient Smoothing (SGS) for LGrad

핵심 아이디어:
한 장에서 gradient를 한 번만 뽑지 말고, 작은 랜덤 변형을 K번 주고
gradient를 K개 뽑아 평균낸 다음 LGrad에 넣는다.

→ 노이즈/압축으로 생기는 고분산 성분을 평균으로 눌러서
   "진짜 남아있는 artifact 성분"을 살린다.

노벨티:
"LGrad의 핵심 표현이 gradient이므로, 입력-공간이 아니라
 gradient-공간에서의 test-time smoothing을 한다"
"""

import copy
from dataclasses import dataclass
from typing import Union, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


@dataclass
class SGSConfig:
    """Configuration for Stochastic Gradient Smoothing"""
    K: int = 5  # Number of views to sample
    augmentation_types: list = None  # ["blur", "denoise", "jpeg", "deblock"]
    blur_sigma_range: tuple = (0.1, 0.5)  # Gaussian blur sigma range
    jpeg_quality_range: tuple = (75, 95)  # JPEG quality range
    denoise_strength: float = 0.02  # Denoising strength
    device: str = "cuda"

    def __post_init__(self):
        if self.augmentation_types is None:
            # Default: mix of blur and JPEG (가장 효과적)
            self.augmentation_types = ["blur", "jpeg"]


class StochasticAugmenter:
    """
    Test-time stochastic augmentation for gradient smoothing.

    매우 약한 변형만 적용 (gradient의 공통 성분을 보존하면서 노이즈만 제거)
    """

    def __init__(self, config: SGSConfig):
        self.cfg = config

    def apply_weak_blur(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply very weak Gaussian blur.

        목적: 고주파 노이즈를 살짝 감소시켜 gradient의 분산 줄이기
        """
        B, C, H, W = x.shape

        # Random sigma for each sample in batch
        sigma = torch.rand(B) * (self.cfg.blur_sigma_range[1] - self.cfg.blur_sigma_range[0])
        sigma = sigma + self.cfg.blur_sigma_range[0]

        # Apply blur (simple average over small kernel)
        kernel_size = 3
        padding = kernel_size // 2

        # Simplified: use average pooling as approximation
        # (더 정확한 Gaussian kernel도 가능하지만 속도 고려)
        x_blur = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)

        # Mix original and blurred (매우 약하게)
        alpha = 0.3  # 30%만 blur
        x_out = alpha * x_blur + (1 - alpha) * x

        return x_out

    def apply_jpeg_recompression(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simulate JPEG recompression with random quality.

        목적: JPEG artifact의 랜덤성을 평균으로 상쇄

        Note: 실제 JPEG 압축은 PIL/OpenCV 필요, 여기서는 간단한 근사 사용
        """
        # JPEG 근사: DCT-based quantization simulation
        # 실전에서는 실제 JPEG encode/decode 사용 권장

        # Simplified approach: add slight quantization noise
        quality = torch.randint(
            self.cfg.jpeg_quality_range[0],
            self.cfg.jpeg_quality_range[1] + 1,
            (1,)
        ).item()

        # Quality를 quantization step으로 변환
        q_step = (100 - quality) / 100.0 * 0.05  # 0~0.05 range

        # Quantization noise
        noise = torch.randn_like(x) * q_step
        x_jpeg = torch.clamp(x + noise, 0, 1)

        return x_jpeg

    def apply_denoise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply very weak denoising (bilateral filter approximation).

        목적: Gaussian noise의 영향을 줄이면서 edge 보존
        """
        # Simple denoising: weighted average with original
        x_smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        # Mix
        alpha = self.cfg.denoise_strength
        x_out = (1 - alpha) * x + alpha * x_smooth

        return x_out

    def apply_deblock(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply deblocking filter (smooth block boundaries).

        목적: JPEG block artifact 경계를 부드럽게
        """
        # Simplified: slight smoothing at 8x8 block boundaries
        # (실제 deblocking은 더 복잡하지만 근사로 충분)

        kernel_size = 5
        padding = kernel_size // 2
        x_smooth = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)

        # Very weak mixing
        alpha = 0.2
        x_out = alpha * x_smooth + (1 - alpha) * x

        return x_out

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentation from configured types.

        Args:
            x: [B, 3, H, W] images in [0, 1]

        Returns:
            x_aug: [B, 3, H, W] augmented images
        """
        # Random choice of augmentation
        aug_type = self.cfg.augmentation_types[
            torch.randint(0, len(self.cfg.augmentation_types), (1,)).item()
        ]

        if aug_type == "blur":
            return self.apply_weak_blur(x)
        elif aug_type == "jpeg":
            return self.apply_jpeg_recompression(x)
        elif aug_type == "denoise":
            return self.apply_denoise(x)
        elif aug_type == "deblock":
            return self.apply_deblock(x)
        else:
            # No augmentation (identity)
            return x


class LGradSGS(nn.Module):
    """
    LGrad with Stochastic Gradient Smoothing (SGS)

    테스트 시 K개의 stochastic view에서 gradient를 추출하고 평균내어
    고분산 노이즈를 제거하고 진짜 artifact 성분만 보존한다.

    Args:
        lgrad_model: Pre-trained LGrad model
        config: SGS configuration

    Example:
        >>> from model.LGrad.lgrad_model import LGrad
        >>> from model.method.sgs import LGradSGS, SGSConfig
        >>>
        >>> # Load base model
        >>> base_lgrad = LGrad(
        ...     stylegan_weights="path/to/stylegan.pth",
        ...     classifier_weights="path/to/classifier.pth",
        ...     device="cuda"
        ... )
        >>>
        >>> # Apply SGS
        >>> config = SGSConfig(K=5, augmentation_types=["blur", "jpeg"], device="cuda")
        >>> model = LGradSGS(base_lgrad, config)
        >>>
        >>> # Use as normal
        >>> model.model.eval()
        >>> predictions = model.predict_proba(images)
    """

    def __init__(self, lgrad_model, config: SGSConfig):
        super().__init__()
        self.cfg = config
        self.device = config.device

        # Deep copy to avoid modifying original model
        self.model = copy.deepcopy(lgrad_model)
        self.model.to(self.device)

        # Augmenter
        self.augmenter = StochasticAugmenter(config)

        print(f"[SGS] Initialized with K={config.K}, augmentations={config.augmentation_types}")

    def extract_smoothed_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract K gradients from stochastic views and average them.

        This is the core of SGS: gradient-space ensemble.

        Args:
            x: Input images [B, 3, H, W], range [0, 1]

        Returns:
            grad_smoothed: [B, 3, 256, 256] averaged gradient
        """
        B = x.shape[0]
        K = self.cfg.K

        # Storage for K gradients per sample
        all_grads = []

        for k in range(K):
            # Apply stochastic augmentation to create view_k
            if k == 0:
                # First view: original (no augmentation)
                x_k = x
            else:
                # Other views: apply random augmentation
                x_k = self.augmenter.augment(x)

            # Extract gradient from this view
            grad_k = self.model.img2grad(x_k)  # [B, 3, 256, 256]

            all_grads.append(grad_k)

        # Stack and average: [K, B, 3, 256, 256] -> [B, 3, 256, 256]
        all_grads = torch.stack(all_grads, dim=0)  # [K, B, 3, 256, 256]
        grad_smoothed = all_grads.mean(dim=0)  # [B, 3, 256, 256]

        return grad_smoothed

    def forward(
        self,
        x: torch.Tensor,
        return_grad: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with Stochastic Gradient Smoothing.

        Args:
            x: Input images [B, 3, H, W], range [0, 1]
            return_grad: If True, also return smoothed gradient

        Returns:
            logits: [B, 1] (positive = fake)
            grad (optional): [B, 3, 256, 256] smoothed gradient
        """
        self.model.eval()

        # Extract smoothed gradient (K-view ensemble)
        grad = self.extract_smoothed_gradient(x)

        # Classify with smoothed gradient
        logits = self.model.classify(grad)

        if return_grad:
            return logits, grad

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict real/fake labels.

        Args:
            x: Input images [B, 3, H, W], range [0, 1]

        Returns:
            predictions: [B] (0=real, 1=fake)
        """
        self.model.eval()
        logits = self.forward(x)
        return (torch.sigmoid(logits) > 0.5).long().squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probability of being fake.

        Args:
            x: Input images [B, 3, H, W], range [0, 1]

        Returns:
            probabilities: [B] (probability of fake)
        """
        self.model.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(1)


# Utility function for easy usage
def create_lgrad_sgs(
    stylegan_weights: str,
    classifier_weights: str,
    K: int = 5,
    augmentation_types: list = None,
    blur_sigma_range: tuple = (0.1, 0.5),
    jpeg_quality_range: tuple = (75, 95),
    device: str = "cuda",
) -> LGradSGS:
    """
    Convenience function to create LGrad model with SGS.

    Args:
        stylegan_weights: Path to StyleGAN discriminator weights
        classifier_weights: Path to ResNet50 classifier weights
        K: Number of stochastic views to ensemble
        augmentation_types: List of augmentations (["blur", "jpeg", "denoise", "deblock"])
        blur_sigma_range: Range of blur sigma
        jpeg_quality_range: Range of JPEG quality for recompression
        device: Device to use

    Returns:
        LGradSGS model ready for inference

    Example:
        >>> model = create_lgrad_sgs(
        ...     stylegan_weights="weights/stylegan.pth",
        ...     classifier_weights="weights/classifier.pth",
        ...     K=5,
        ...     augmentation_types=["blur", "jpeg"]
        ... )
        >>> probs = model.predict_proba(corrupted_images)
    """
    from model.LGrad.lgrad_model import LGrad

    # Create base LGrad model
    lgrad = LGrad(
        stylegan_weights=stylegan_weights,
        classifier_weights=classifier_weights,
        device=device,
        resize=256,
    )

    # Apply SGS
    if augmentation_types is None:
        augmentation_types = ["blur", "jpeg"]

    config = SGSConfig(
        K=K,
        augmentation_types=augmentation_types,
        blur_sigma_range=blur_sigma_range,
        jpeg_quality_range=jpeg_quality_range,
        device=device,
    )

    model = LGradSGS(lgrad, config)

    return model
