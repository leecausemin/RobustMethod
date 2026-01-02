"""
Structure-Aware Sharpening (SAS) for robust deepfake detection

핵심 아이디어:
corruption과 deepfake artifact를 구분하여 진짜 deepfake artifact만 강조한다.
- Consistency: Multi-view에서 일관된(분산 낮음) 성분만 유지
- Coherence: 구조적/방향성 있는(edge-like) 패턴만 유지
→ Random/block corruption은 억제, 구조적 deepfake artifact는 강조

방법론:
1. K개의 denoised view에서 artifact 추출
2. Median artifact 계산 (outlier에 robust)
3. Consistency mask: 분산(or MAD)이 낮은 픽셀만 통과
4. Coherence mask: 구조적 방향성이 있는 픽셀만 통과
5. 두 mask를 곱해서 최종 artifact 강조
"""

import copy
import math
from dataclasses import dataclass
from typing import Union, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SASConfig:
    """Configuration for Structure-Aware Sharpening (SAS)"""
    K: int = 5  # Number of views to sample

    # Model type
    model: Literal["LGrad", "NPR"] = "LGrad"

    # Huber-TV 적용 대상
    denoise_target: Literal["input", "artifact"] = "artifact"

    # 기본 Huber-TV 하이퍼파라미터
    huber_tv_lambda: float = 0.05
    huber_tv_delta: float = 0.01
    huber_tv_iterations: int = 3
    huber_tv_step_size: float = 0.2

    # 파라미터 생성 전략
    lambda_range: tuple[float, float] = (0.5, 1.5)
    jitter_strength: float = 0.1
    couple_delta_to_lambda: bool = True

    # K개의 파라미터 세트 (None이면 자동 생성)
    param_sets: Optional[list[tuple[float, float, int, float]]] = None

    # === SAS 전용 파라미터 ===

    # Consistency mask 설정
    consistency_metric: Literal["variance", "mad"] = "mad"
    mask_type: Literal["soft", "hard"] = "soft"
    tau: float = 0.1  # Hard mask threshold
    gamma: float = 10.0  # Soft mask sharpness

    # Coherence mask 설정
    coherence_method: Literal["structure_tensor", "gradient_mag"] = "structure_tensor"
    eta: float = 1.0  # Coherence power (1~2 추천)
    gaussian_sigma: float = 1.0  # Structure tensor smoothing
    coherence_eps: float = 0.4  # Soft suppression 바닥값 (0.3~0.6)

    # Blending 설정 (GPT v3 개선)
    blend_min: float = 0.3  # Minimum blending ratio (보수적)
    blend_max: float = 1.0  # Maximum blending ratio
    smooth_sigma: float = 3.0  # Artifact smoothing kernel size
    guide_blur_sigma: float = 1.5  # Guide image pre-blur for stronger coherence

    # Robust fusion 설정
    fusion_scale: float = 2.0  # c in exp(-d_k / (c * mad)), 1~3 추천

    # GPT 패치 (C): Mean + Robust hybrid
    fusion_alpha: float = 0.3  # Robust 비중 (0=mean only, 1=robust only)
                                # 0.2~0.4 추천 (mean이 신호 보존, robust가 노이즈 억제)

    device: str = "cuda"

    def __post_init__(self):
        """K개의 다양한 파라미터 세트를 자동 생성"""
        if self.param_sets is None:
            self.param_sets = self._generate_param_sets()

        if len(self.param_sets) != self.K:
            raise ValueError(f"param_sets length ({len(self.param_sets)}) must match K ({self.K})")

    def _generate_param_sets(self) -> list[tuple[float, float, int, float]]:
        """K개의 다양한 하이퍼파라미터 세트를 자동 생성"""
        import numpy as np

        param_sets = []
        base_lambda = self.huber_tv_lambda
        base_delta = self.huber_tv_delta

        # View 0: Original (no denoising)
        param_sets.append((0.0, 0.0, 0, 0.0))

        # View 1~K-1: 좁은 범위에서 균등 간격 + jitter
        if self.K > 1:
            lambda_min = base_lambda * self.lambda_range[0]
            lambda_max = base_lambda * self.lambda_range[1]

            for i in range(self.K - 1):
                # 균등 간격
                lambda_base = lambda_min + (lambda_max - lambda_min) * i / max(1, self.K - 2)

                # 작은 jitter 추가
                jitter = np.random.uniform(-self.jitter_strength, self.jitter_strength)
                lambda_i = lambda_base * (1 + jitter)
                lambda_i = max(0.0, lambda_i)

                # δ를 λ와 함께 움직임
                if self.couple_delta_to_lambda:
                    delta_ratio = np.sqrt(lambda_i / (base_lambda + 1e-8))
                    delta_i = base_delta * delta_ratio
                else:
                    delta_i = base_delta

                param_sets.append((
                    lambda_i,
                    delta_i,
                    self.huber_tv_iterations,
                    self.huber_tv_step_size
                ))

        return param_sets


class StochasticAugmenter:
    """Test-time Anisotropic Huber-TV augmentation"""

    def __init__(self, config: SASConfig):
        self.cfg = config

    def apply_anisotropic_huber_tv(
        self,
        x: torch.Tensor,
        lambda_tv: float = None,
        huber_delta: float = None,
        iterations: int = None,
        step_size: float = None,
        skip_clamp: bool = False
    ) -> torch.Tensor:
        """
        Apply Anisotropic Huber Total Variation denoising

        Args:
            skip_clamp: If True, skip clamping (for artifact denoising)
        """
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
            # Compute spatial gradients
            dx = torch.zeros_like(x_denoise)
            dx[:, :, :, :-1] = x_denoise[:, :, :, 1:] - x_denoise[:, :, :, :-1]

            dy = torch.zeros_like(x_denoise)
            dy[:, :, :-1, :] = x_denoise[:, :, 1:, :] - x_denoise[:, :, :-1, :]

            # Huber function derivative
            def huber_grad(t, delta):
                return t / (delta + torch.abs(t))

            gx = huber_grad(dx, huber_delta)
            gy = huber_grad(dy, huber_delta)

            # Divergence
            div_x = torch.zeros_like(x_denoise)
            div_x[:, :, :, 1:] = gx[:, :, :, 1:] - gx[:, :, :, :-1]
            div_x[:, :, :, 0] = gx[:, :, :, 0]

            div_y = torch.zeros_like(x_denoise)
            div_y[:, :, 1:, :] = gy[:, :, 1:, :] - gy[:, :, :-1, :]
            div_y[:, :, 0, :] = gy[:, :, 0, :]

            # Gradient descent step
            x_denoise = x_denoise - step_size * (
                (x_denoise - x) - lambda_tv * (div_x + div_y)
            )

            # GPT 패치 (A): Clamp only for input images, not for artifacts
            if not skip_clamp:
                x_denoise = torch.clamp(x_denoise, 0, 1)

        return x_denoise


class UnifiedSAS(nn.Module):
    """
    Unified Structure-Aware Sharpening (SAS) for both LGrad and NPR

    일관성과 구조성 기반으로 진짜 deepfake artifact만 선택적으로 강조
    """

    def __init__(self, base_model, config: SASConfig):
        super().__init__()
        self.cfg = config
        self.device = config.device

        # Deep copy to avoid modifying original model
        self.model = copy.deepcopy(base_model)
        self.model.to(self.device)

        # Augmenter
        self.augmenter = StochasticAugmenter(config)

        # Validate config
        if config.model not in ["LGrad", "NPR"]:
            raise ValueError(f"Unsupported model type: {config.model}")

        # Pre-compute Gaussian kernel for structure tensor
        if config.coherence_method == "structure_tensor":
            self.gaussian_kernel = self._create_gaussian_kernel(
                sigma=config.gaussian_sigma,
                device=self.device
            )

        print(f"[SAS] Initialized for {config.model} with K={config.K}, denoise_target={config.denoise_target}")
        print(f"[SAS] Base params: λ={config.huber_tv_lambda}, δ={config.huber_tv_delta}")
        print(f"[SAS] Consistency: metric={config.consistency_metric}, mask_type={config.mask_type}")
        print(f"[SAS] Coherence: method={config.coherence_method}, eta={config.eta}")
        print(f"[SAS] Parameter sets for {config.K} views:")
        for k, (lam, delta, iters, step) in enumerate(config.param_sets):
            print(f"  View {k}: λ={lam:.4f}, δ={delta:.4f}, iter={iters}, step={step:.2f}")

    def _create_gaussian_kernel(
        self,
        sigma: float,
        device: str
    ) -> torch.Tensor:
        """Create 2D Gaussian kernel for smoothing structure tensor"""
        # Kernel size: k = 2 * ceil(3 * sigma) + 1
        kernel_size = 2 * math.ceil(3 * sigma) + 1

        # Create 1D Gaussian
        x = torch.arange(kernel_size, dtype=torch.float32, device=device)
        x = x - kernel_size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()

        # Create 2D Gaussian (outer product)
        kernel = gauss_1d[:, None] * gauss_1d[None, :]
        kernel = kernel / kernel.sum()

        # Reshape for conv2d: [1, 1, K, K]
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        return kernel

    def extract_multiview_artifacts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract K artifacts from K denoised views

        Returns:
            artifacts_stack: [K, B, C, H, W] K개의 artifact
            views_stack: [K, B, C, H, W] K개의 view images (for coherence)
        """
        B = x.shape[0]
        K = self.cfg.K

        all_artifacts = []
        all_views = []

        if self.cfg.denoise_target == "artifact":
            # Extract artifact once, then denoise it K times
            if self.cfg.model == "LGrad":
                artifact = self.model.img2grad(x)
            else:  # NPR
                artifact = self.model.img2npr(x)

            # GPT 개선 (1): Artifact 스케일 정규화
            # Per-sample robust scale (MAD or median of absolute values)
            # Shape: [B, 1, 1, 1] for broadcasting
            artifact_scale = artifact.abs().reshape(B, -1).median(dim=1)[0]
            artifact_scale = artifact_scale.view(B, 1, 1, 1) + 1e-8  # Avoid division by zero

            for k in range(K):
                lambda_tv, delta, iterations, step_size = self.cfg.param_sets[k]

                if k == 0:
                    # View 0: original (no denoising)
                    artifact_k = artifact
                else:
                    # Normalize artifact by scale
                    artifact_norm = artifact / artifact_scale

                    # Apply Huber-TV to normalized artifact (skip clamp to preserve distribution)
                    artifact_denoised_norm = self.augmenter.apply_anisotropic_huber_tv(
                        artifact_norm,
                        lambda_tv=lambda_tv,
                        huber_delta=delta,
                        iterations=iterations,
                        step_size=step_size,
                        skip_clamp=True  # GPT 패치 (A): artifact는 clamp 안 함
                    )

                    # Denormalize back to original scale
                    artifact_k = artifact_denoised_norm * artifact_scale

                all_artifacts.append(artifact_k)
                all_views.append(x)  # Original x for coherence

        else:  # denoise_target == "input"
            # Denoise input K times, then extract artifact
            for k in range(K):
                lambda_tv, delta, iterations, step_size = self.cfg.param_sets[k]

                if k == 0:
                    # View 0: original (no denoising)
                    x_k = x
                else:
                    # Apply Huber-TV to input
                    x_k = self.augmenter.apply_anisotropic_huber_tv(
                        x,
                        lambda_tv=lambda_tv,
                        huber_delta=delta,
                        iterations=iterations,
                        step_size=step_size
                    )

                # Extract artifact from denoised input
                if self.cfg.model == "LGrad":
                    artifact_k = self.model.img2grad(x_k)
                else:  # NPR
                    artifact_k = self.model.img2npr(x_k)

                all_artifacts.append(artifact_k)
                all_views.append(x_k)  # Denoised view for coherence

        # Stack: [K, B, C, H, W]
        artifacts_stack = torch.stack(all_artifacts, dim=0)
        views_stack = torch.stack(all_views, dim=0)

        return artifacts_stack, views_stack

    def compute_robust_fusion(
        self,
        artifacts_stack: torch.Tensor
    ) -> torch.Tensor:
        """
        Robust fusion of multi-view artifacts using weighted average.

        GPT 핵심 개선:
        - "Mask 곱하기 (삭제)" → "가중 평균 (복원)"
        - 각 view의 신뢰도로 가중치를 주어 robust하게 합침

        Args:
            artifacts_stack: [K, B, C, H, W] K개의 artifact

        Returns:
            artifact_fused: [B, C, H, W] 복원된 artifact
        """
        K, B, C, H, W = artifacts_stack.shape
        eps = 1e-8

        # Step 1: Compute median artifact (robust center)
        median_artifact = torch.median(artifacts_stack, dim=0)[0]  # [B, C, H, W]

        # Step 2: Compute per-view deviation from median
        # d_k = |a_k - a_med|
        deviations = torch.abs(artifacts_stack - median_artifact.unsqueeze(0))  # [K, B, C, H, W]

        # Step 3: Compute MAD (Median Absolute Deviation) for normalization
        # MAD = median_k(d_k)
        mad = torch.median(deviations, dim=0)[0]  # [B, C, H, W]

        # GPT 패치 (B): MAD floor to prevent weight explosion
        # When MAD is too small, weights can explode and select only one view
        global_mad = mad.mean()  # Global reference
        mad_floor = 0.1 * global_mad  # 10% of global MAD as minimum
        mad = torch.maximum(mad, mad_floor)

        # Step 4: Compute per-view weights
        # w_k = exp(-d_k / (c * MAD))
        # c 클수록 모든 view에 비슷한 가중치 (보수적)
        # c 작을수록 median에 가까운 view만 높은 가중치 (공격적)
        c = self.cfg.fusion_scale
        weights = torch.exp(-deviations / (c * mad.unsqueeze(0)))  # [K, B, C, H, W]

        # Step 5: Weighted average (robust fusion)
        # a_hat = Σ(w_k * a_k) / Σ(w_k)
        weighted_sum = (weights * artifacts_stack).sum(dim=0)  # [B, C, H, W]
        weight_sum = weights.sum(dim=0)  # [B, C, H, W]
        artifact_fused = weighted_sum / (weight_sum + eps)  # [B, C, H, W]

        return artifact_fused

    def apply_edge_preserving_smoothing(self, artifact: torch.Tensor) -> torch.Tensor:
        """
        Apply edge-preserving smoothing to artifact.

        GPT v3 개선: Gaussian blur로 smooth version 생성
        나중에 coherence로 sharp/smooth를 블렌딩

        Args:
            artifact: [B, C, H, W] artifact to smooth

        Returns:
            artifact_smooth: [B, C, H, W] smoothed artifact
        """
        B, C, H, W = artifact.shape

        # Create Gaussian kernel for smoothing
        sigma = self.cfg.smooth_sigma
        kernel_size = 2 * math.ceil(3 * sigma) + 1

        # 1D Gaussian - ensure same device and dtype as artifact
        x = torch.arange(kernel_size, dtype=artifact.dtype, device=artifact.device)
        x = x - kernel_size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()

        # 2D Gaussian
        kernel = gauss_1d[:, None] * gauss_1d[None, :]
        kernel = kernel / kernel.sum()

        # Reshape kernel for conv2d
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        # Apply per-channel (C개 채널 각각)
        artifact_smooth = torch.zeros_like(artifact)
        for c in range(C):
            artifact_c = artifact[:, c:c+1, :, :]  # [B, 1, H, W]
            artifact_c_smooth = F.conv2d(
                artifact_c,
                kernel,  # Already on correct device and dtype
                padding=kernel_size // 2
            )
            artifact_smooth[:, c:c+1, :, :] = artifact_c_smooth

        return artifact_smooth

    def compute_coherence_mask(self, reference_image: torch.Tensor) -> torch.Tensor:
        """
        Compute coherence mask from reference image (NOT artifact!).

        구조적 방향성이 있는(edge-like) 픽셀만 통과시킴.
        Random noise: 방향성 없음 → coherence 낮음
        Deepfake artifact (얼굴 경계, 합성 경계): 방향성 있음 → coherence 높음

        GPT 개선사항:
        - median_artifact가 아니라 median(x_k) 또는 원본 x에서 계산
        - Coherence는 "어디가 구조적인 위치냐"를 알려주는 spatial prior
        - GPT v3: guide image에 pre-blur 추가로 더 강건한 coherence 계산
        """
        B, C, H, W = reference_image.shape

        # Convert to grayscale
        gray = reference_image.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # GPT v3 개선: Pre-blur guide image for stronger coherence
        if self.cfg.guide_blur_sigma > 0:
            sigma = self.cfg.guide_blur_sigma
            kernel_size = 2 * math.ceil(3 * sigma) + 1

            # Create Gaussian kernel
            x = torch.arange(kernel_size, dtype=gray.dtype, device=gray.device)
            x = x - kernel_size // 2
            gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
            gauss_1d = gauss_1d / gauss_1d.sum()

            kernel = gauss_1d[:, None] * gauss_1d[None, :]
            kernel = kernel / kernel.sum()

            # Apply blur
            gray = F.conv2d(
                gray,
                kernel.view(1, 1, kernel_size, kernel_size),
                padding=kernel_size // 2
            )

        if self.cfg.coherence_method == "gradient_mag":
            # Simple method: Use gradient magnitude
            sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=gray.dtype, device=gray.device
            ).view(1, 1, 3, 3)
            sobel_y = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=gray.dtype, device=gray.device
            ).view(1, 1, 3, 3)

            grad_x = F.conv2d(gray, sobel_x, padding=1)
            grad_y = F.conv2d(gray, sobel_y, padding=1)

            grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

            # Normalize to [0, 1]
            grad_mag = grad_mag / (grad_mag.max() + 1e-8)

            # Apply power
            M_coh = grad_mag ** self.cfg.eta

        else:  # structure_tensor
            # Structure tensor method (권장)
            # Step 1: Compute gradients
            sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=gray.dtype, device=gray.device
            ).view(1, 1, 3, 3)
            sobel_y = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=gray.dtype, device=gray.device
            ).view(1, 1, 3, 3)

            grad_x = F.conv2d(gray, sobel_x, padding=1)
            grad_y = F.conv2d(gray, sobel_y, padding=1)

            # Step 2: Compute structure tensor components
            Jxx = grad_x * grad_x
            Jyy = grad_y * grad_y
            Jxy = grad_x * grad_y

            # Step 3: Apply Gaussian smoothing (필수!)
            gaussian_kernel = self.gaussian_kernel.to(dtype=gray.dtype, device=gray.device)

            Jxx_smooth = F.conv2d(Jxx, gaussian_kernel, padding=gaussian_kernel.shape[-1]//2)
            Jyy_smooth = F.conv2d(Jyy, gaussian_kernel, padding=gaussian_kernel.shape[-1]//2)
            Jxy_smooth = F.conv2d(Jxy, gaussian_kernel, padding=gaussian_kernel.shape[-1]//2)

            # Step 4: Compute eigenvalues
            trace = Jxx_smooth + Jyy_smooth
            det = Jxx_smooth * Jyy_smooth - Jxy_smooth * Jxy_smooth

            eps = 1e-8
            discriminant = torch.clamp(trace**2 / 4 - det, min=0)

            lambda1 = trace / 2 + torch.sqrt(discriminant + eps)
            lambda2 = trace / 2 - torch.sqrt(discriminant + eps)

            # Step 5: Compute coherence
            # coherence = (λ1 - λ2)^2 / (λ1 + λ2)^2
            coherence = (lambda1 - lambda2)**2 / (lambda1 + lambda2 + eps)**2

            # Apply power (eta) - 보통 1~2 정도면 충분
            M_coh = coherence ** self.cfg.eta

            # GPT v3 개선: Blending ratio로 사용 (clamp to [blend_min, blend_max])
            # 이전: soft suppression (eps + (1-eps)*coherence) → multiplication
            # 지금: blending ratio → m * sharp + (1-m) * smooth

        # Clamp to [blend_min, blend_max] for conservative blending
        M_coh = torch.clamp(M_coh, self.cfg.blend_min, self.cfg.blend_max)

        return M_coh  # [B, 1, H, W]

    def forward(
        self,
        x: torch.Tensor,
        return_artifact: bool = False,
        return_masks: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass with Structure-Aware Sharpening

        핵심: Robust Fusion만 사용 (Coherence mask 제거)
        - Multi-view artifacts 생성
        - Robust fusion으로 가중 평균
        - Fused artifact 직접 사용
        """
        self.model.eval()

        # Step 1: Extract K artifacts from multi-view + views
        artifacts_stack, views_stack = self.extract_multiview_artifacts(x)

        # Step 2: Compute both mean and robust fusion
        artifact_mean = artifacts_stack.mean(dim=0)  # [B, C, H, W] - 신호 보존
        artifact_robust = self.compute_robust_fusion(artifacts_stack)  # [B, C, H, W] - 노이즈 억제

        # GPT 패치 (C): Mean + Robust hybrid
        # Mean: preserves subtle signals
        # Robust: removes outlier noise
        alpha = self.cfg.fusion_alpha
        artifact_enhanced = (1 - alpha) * artifact_mean + alpha * artifact_robust  # [B, C, H, W]

        # Bonus: Distribution matching to original artifact (maintain classifier expectations)
        # Match mean and std of fused artifact to original artifact (view 0)
        artifact_original = artifacts_stack[0]  # [B, C, H, W] - original artifact (no denoising)

        # Per-sample affine matching
        mean_orig = artifact_original.mean(dim=(1, 2, 3), keepdim=True)
        std_orig = artifact_original.std(dim=(1, 2, 3), keepdim=True)
        mean_fused = artifact_enhanced.mean(dim=(1, 2, 3), keepdim=True)
        std_fused = artifact_enhanced.std(dim=(1, 2, 3), keepdim=True)

        artifact_enhanced = (artifact_enhanced - mean_fused) / (std_fused + 1e-8) * std_orig + mean_orig

        # Step 3: Classify enhanced artifact
        logits = self.model.classify(artifact_enhanced)

        # Return based on flags
        returns = [logits]
        if return_artifact:
            returns.append(artifact_enhanced)
        if return_masks:
            # For debugging: return mean, robust, and final hybrid
            masks = {
                'artifact_fused': artifact_enhanced,  # Final hybrid result
                'artifact_mean': artifact_mean,       # Simple mean
                'artifact_robust': artifact_robust,   # Robust fusion
            }
            returns.append(masks)

        if len(returns) == 1:
            return returns[0]
        else:
            return tuple(returns)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict real/fake labels"""
        self.model.eval()
        logits = self.forward(x)
        return (torch.sigmoid(logits) > 0.5).long().squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probability of being fake"""
        self.model.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(1)
