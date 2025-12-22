"""
Stochastic Gradient Smoothing (SGS) for LGrad using Anisotropic Huber-TV

핵심 아이디어:
한 장에서 gradient를 한 번만 뽑지 말고, Anisotropic Huber-TV denoising을
K번 적용해서 gradient를 K개 뽑아 평균낸 다음 LGrad에 넣는다.

→ Huber-TV의 edge-preserving denoising을 통해 Gaussian noise를 제거하면서
   노이즈/압축으로 생기는 고분산 성분을 평균으로 눌러서
   "진짜 남아있는 artifact 성분"을 살린다.

노벨티:
1. LGrad의 핵심 표현이 gradient이므로, 입력-공간이 아니라
   gradient-공간에서의 test-time smoothing을 한다
2. Anisotropic Huber-TV를 사용하여 edge는 보존하면서 smooth region의 노이즈만 제거
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
    """Configuration for Stochastic Gradient Smoothing with Huber-TV

    Note: lambda_tv의 적절한 값은 gradient 스케일에 의존합니다.
    현재 기본값은 LGrad의 gradient map 스케일 기준입니다.

    K개의 view를 생성할 때, 각기 다른 하이퍼파라미터를 사용하여
    다양한 노이즈 타입(gaussian, jpeg, blur 등)에 robust하게 대응합니다.
    """
    K: int = 5  # Number of views to sample

    # Huber-TV 적용 대상
    denoise_target: Literal["input", "gradient"] = "gradient"
    # "input": 원본 이미지에 Huber-TV 적용 후 gradient 추출
    # "gradient": gradient 추출 후 Huber-TV 적용 (기본값)

    # 기본 하이퍼파라미터 (K개의 파라미터 세트를 자동 생성할 때 사용)
    huber_tv_lambda: float = 0.05  # TV regularization strength
    huber_tv_delta: float = 0.01  # Huber threshold (L1/L2 transition point)
    huber_tv_iterations: int = 3  # Number of gradient descent iterations
    huber_tv_step_size: float = 0.2  # Gradient descent step size

    # 파라미터 생성 전략
    lambda_range: tuple[float, float] = (0.5, 1.5)  # base 대비 범위 (좁은 범위)
    jitter_strength: float = 0.1  # 랜덤 jitter 강도 (0.1 = ±10%)
    couple_delta_to_lambda: bool = True  # δ를 λ와 함께 움직일지

    # K개의 다양한 파라미터 세트 (None이면 자동 생성)
    # 각 원소는 (lambda_tv, delta, iterations, step_size) 튜플
    param_sets: Optional[list[tuple[float, float, int, float]]] = None

    device: str = "cuda"

    def __post_init__(self):
        """K개의 다양한 파라미터 세트를 자동 생성"""
        if self.param_sets is None:
            self.param_sets = self._generate_param_sets()

        # K와 param_sets 길이가 맞는지 확인
        if len(self.param_sets) != self.K:
            raise ValueError(f"param_sets length ({len(self.param_sets)}) must match K ({self.K})")

    def _generate_param_sets(self) -> list[tuple[float, float, int, float]]:
        """
        K개의 다양한 하이퍼파라미터 세트를 자동 생성.

        전략: D(좁은 범위) + 약한 C(jitter)
        - 기본값 주변 좁은 범위(±50%)에서 균등 간격
        - δ를 λ와 함께 움직임: δ ∝ √(λ/base_λ)
        - 각 값에 작은 랜덤 jitter (±10%) 추가

        이유:
        - 너무 넓은 범위: 일부 view가 구조를 과하게 깎아 평균 망가짐
        - 너무 좁은 범위: view들이 비슷해서 평균의 이점 감소
        - δ를 함께 움직임: λ만 키우면 구조 손상, δ도 키우면 부드럽게 조절

        Returns:
            K개의 (lambda_tv, delta, iterations, step_size) 튜플 리스트
        """
        import numpy as np

        param_sets = []
        base_lambda = self.huber_tv_lambda
        base_delta = self.huber_tv_delta

        # View 0: Original (no denoising)
        param_sets.append((0.0, 0.0, 0, 0.0))

        # View 1~K-1: 좁은 범위에서 균등 간격 + jitter
        if self.K > 1:
            # λ 범위: [0.5*base, 1.5*base] 에서 균등 간격
            lambda_min = base_lambda * self.lambda_range[0]
            lambda_max = base_lambda * self.lambda_range[1]

            for i in range(self.K - 1):
                # 균등 간격
                lambda_base = lambda_min + (lambda_max - lambda_min) * i / max(1, self.K - 2)

                # 작은 jitter 추가 (±10%)
                jitter = np.random.uniform(-self.jitter_strength, self.jitter_strength)
                lambda_i = lambda_base * (1 + jitter)
                lambda_i = max(0.0, lambda_i)  # 음수 방지

                # δ를 λ와 함께 움직임
                if self.couple_delta_to_lambda:
                    # δ ∝ √(λ/base_λ) (부드럽게 조절)
                    delta_ratio = np.sqrt(lambda_i / (base_lambda + 1e-8))
                    delta_i = base_delta * delta_ratio
                else:
                    # δ는 고정 (옛날 방식)
                    delta_i = base_delta

                # Iterations와 step_size는 고정
                param_sets.append((
                    lambda_i,
                    delta_i,
                    self.huber_tv_iterations,
                    self.huber_tv_step_size
                ))

        return param_sets


class StochasticAugmenter:
    """
    Test-time Anisotropic Huber-TV augmentation for gradient smoothing.

    Huber-TV를 사용한 edge-preserving denoising으로 gradient의 공통 성분을 보존하면서 노이즈만 제거
    """

    def __init__(self, config: SGSConfig):
        self.cfg = config

    def apply_anisotropic_huber_tv(
        self,
        x: torch.Tensor,
        lambda_tv: float = None,
        huber_delta: float = None,
        iterations: int = None,
        step_size: float = None
    ) -> torch.Tensor:
        """
        Apply Anisotropic Huber Total Variation denoising.

        목적: Edge-preserving denoising (Gaussian noise 제거에 특히 효과적)

        Anisotropic TV:
        - 수평/수직 방향을 독립적으로 처리
        - Edge는 보존하면서 smooth region의 노이즈만 제거

        Huber norm:
        - L1 (robust to outliers) + L2 (smooth) 혼합
        - δ 작음: outlier/블록/스파이크에 robust, edge 보존↑
        - δ 큼: 전반적으로 더 매끈하지만 edge도 같이 깎일 수 있음

        Args:
            x: Input image [B, C, H, W]
            lambda_tv: TV regularization strength (작을수록 약한 denoising)
            huber_delta: Huber threshold (작을수록 L1 성향↑, outlier에 robust)
            iterations: Number of gradient descent steps
            step_size: Gradient descent step size

        Returns:
            Denoised image [B, C, H, W]
        """
        # Use config values if not specified
        if lambda_tv is None:
            lambda_tv = self.cfg.huber_tv_lambda
        if huber_delta is None:
            huber_delta = self.cfg.huber_tv_delta
        if iterations is None:
            iterations = self.cfg.huber_tv_iterations
        if step_size is None:
            step_size = self.cfg.huber_tv_step_size

        x_denoise = x.clone()

        for _ in range(iterations):
            # Compute spatial gradients (finite differences)
            # Horizontal gradient
            dx = torch.zeros_like(x_denoise)
            dx[:, :, :, :-1] = x_denoise[:, :, :, 1:] - x_denoise[:, :, :, :-1]

            # Vertical gradient
            dy = torch.zeros_like(x_denoise)
            dy[:, :, :-1, :] = x_denoise[:, :, 1:, :] - x_denoise[:, :, :-1, :]

            # Huber function derivative (soft thresholding)
            # d/dt huber(t) = t / delta      if |t| <= delta
            #               = sign(t)        if |t| > delta
            def huber_grad(t, delta):
                # Smooth approximation
                return t / (delta + torch.abs(t))

            # Compute Huber TV gradient
            gx = huber_grad(dx, huber_delta)
            gy = huber_grad(dy, huber_delta)

            # Divergence (negative gradient of TV)
            div_x = torch.zeros_like(x_denoise)
            div_x[:, :, :, 1:] = gx[:, :, :, 1:] - gx[:, :, :, :-1]
            div_x[:, :, :, 0] = gx[:, :, :, 0]

            div_y = torch.zeros_like(x_denoise)
            div_y[:, :, 1:, :] = gy[:, :, 1:, :] - gy[:, :, :-1, :]
            div_y[:, :, 0, :] = gy[:, :, 0, :]

            # Gradient descent step
            # x_{k+1} = x_k - step_size * [x_k - x_0 - lambda * div(grad_TV)]
            x_denoise = x_denoise - step_size * (
                (x_denoise - x) - lambda_tv * (div_x + div_y)
            )

            # Clamp to valid range
            x_denoise = torch.clamp(x_denoise, 0, 1)

        return x_denoise

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Anisotropic Huber-TV denoising.

        Args:
            x: [B, 3, H, W] images in [0, 1]

        Returns:
            x_aug: [B, 3, H, W] denoised images
        """
        return self.apply_anisotropic_huber_tv(x)


class LGradSGS(nn.Module):
    """
    LGrad with Stochastic Gradient Smoothing (SGS) using Anisotropic Huber-TV

    테스트 시 K개의 Huber-TV denoised view에서 gradient를 추출하고 평균내어
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
        >>> # Apply SGS with Huber-TV
        >>> config = SGSConfig(
        ...     K=5,
        ...     huber_tv_lambda=0.05,
        ...     huber_tv_delta=0.01,
        ...     huber_tv_iterations=3,
        ...     huber_tv_step_size=0.2,
        ...     device="cuda"
        ... )
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

        print(f"[SGS] Initialized with K={config.K}, denoise_target={config.denoise_target}")
        print(f"[SGS] Base params: λ={config.huber_tv_lambda}, δ={config.huber_tv_delta}, iter={config.huber_tv_iterations}, step={config.huber_tv_step_size}")
        print(f"[SGS] Parameter sets for {config.K} views:")
        for k, (lam, delta, iters, step) in enumerate(config.param_sets):
            print(f"  View {k}: λ={lam:.4f}, δ={delta:.4f}, iter={iters}, step={step:.2f}")

    def extract_smoothed_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract gradient and apply K different Huber-TV denoising, then average.

        This is the core of SGS: gradient-space ensemble with edge-preserving denoising.

        동작 방식 (denoise_target에 따라):
        - "gradient" (기본값):
            1. 입력 이미지 → gradient 추출 (한 번만)
            2. Gradient에 K개의 다른 Huber-TV 파라미터 적용
            3. K개의 denoised gradient를 평균
        - "input":
            1. 입력 이미지에 K개의 다른 Huber-TV 파라미터 적용
            2. 각 denoised 이미지 → gradient 추출
            3. K개의 gradient를 평균

        Args:
            x: Input images [B, 3, H, W], range [0, 1]

        Returns:
            grad_smoothed: [B, 3, 256, 256] averaged gradient
        """
        B = x.shape[0]
        K = self.cfg.K

        all_grads = []

        if self.cfg.denoise_target == "gradient":
            # Step 1: Extract gradient from original image (only once!)
            grad = self.model.img2grad(x)  # [B, 3, 256, 256]

            # Step 2: Apply K different Huber-TV denoising to the gradient
            for k in range(K):
                # Get parameters for this view
                lambda_tv, delta, iterations, step_size = self.cfg.param_sets[k]

                # Apply Huber-TV denoising to gradient
                if k == 0:
                    # First view: original gradient (no denoising)
                    grad_k = grad
                else:
                    # Apply Huber-TV denoising to gradient with view-specific parameters
                    grad_k = self.augmenter.apply_anisotropic_huber_tv(
                        grad,
                        lambda_tv=lambda_tv,
                        huber_delta=delta,
                        iterations=iterations,
                        step_size=step_size
                    )

                all_grads.append(grad_k)

        else:  # denoise_target == "input"
            # Step 1: Apply K different Huber-TV denoising to input image
            for k in range(K):
                # Get parameters for this view
                lambda_tv, delta, iterations, step_size = self.cfg.param_sets[k]

                # Apply Huber-TV denoising to input
                if k == 0:
                    # First view: original input (no denoising)
                    x_k = x
                else:
                    # Apply Huber-TV denoising to input with view-specific parameters
                    x_k = self.augmenter.apply_anisotropic_huber_tv(
                        x,
                        lambda_tv=lambda_tv,
                        huber_delta=delta,
                        iterations=iterations,
                        step_size=step_size
                    )

                # Step 2: Extract gradient from denoised input
                grad_k = self.model.img2grad(x_k)

                all_grads.append(grad_k)

        # Step 3: Average K gradients
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
    denoise_target: Literal["input", "gradient"] = "gradient",
    huber_tv_lambda: float = 0.05,
    huber_tv_delta: float = 0.01,
    huber_tv_iterations: int = 3,
    huber_tv_step_size: float = 0.2,
    device: str = "cuda",
) -> LGradSGS:
    """
    Convenience function to create LGrad model with SGS using Huber-TV.

    Note: 파라미터 값들은 gradient 스케일에 의존합니다.
    현재 기본값은 LGrad의 gradient map 기준입니다.

    Args:
        stylegan_weights: Path to StyleGAN discriminator weights
        classifier_weights: Path to ResNet50 classifier weights
        K: Number of stochastic views to ensemble
        denoise_target: Where to apply Huber-TV ("input" or "gradient")
        huber_tv_lambda: TV regularization strength
        huber_tv_delta: Huber threshold (작을수록 L1 성향↑, outlier에 robust)
        huber_tv_iterations: Number of gradient descent iterations
        huber_tv_step_size: Gradient descent step size
        device: Device to use

    Returns:
        LGradSGS model ready for inference

    Example:
        >>> # Denoise gradient (기본값)
        >>> model = create_lgrad_sgs(
        ...     stylegan_weights="weights/stylegan.pth",
        ...     classifier_weights="weights/classifier.pth",
        ...     K=5,
        ...     denoise_target="gradient",
        ...     huber_tv_lambda=0.05
        ... )
        >>>
        >>> # Denoise input image
        >>> model = create_lgrad_sgs(
        ...     stylegan_weights="weights/stylegan.pth",
        ...     classifier_weights="weights/classifier.pth",
        ...     K=5,
        ...     denoise_target="input",
        ...     huber_tv_lambda=0.05
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

    # Apply SGS with Huber-TV
    config = SGSConfig(
        K=K,
        denoise_target=denoise_target,
        huber_tv_lambda=huber_tv_lambda,
        huber_tv_delta=huber_tv_delta,
        huber_tv_iterations=huber_tv_iterations,
        huber_tv_step_size=huber_tv_step_size,
        device=device,
    )

    model = LGradSGS(lgrad, config)

    return model
