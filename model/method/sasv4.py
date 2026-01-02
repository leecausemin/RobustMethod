"""
SAS v4 (Structure-Aware Sharpening v4) - Residual-only with Disagreement Mask

핵심 아이디어:
원본 artifact를 anchor로 유지하고, multi-view disagreement가 큰 픽셀에서만
residual correction을 적용한다.

수식:
- a0 = 원본 artifact (view 0)
- af = multi-view fused artifact
- r = af - a0 (residual)
- m = disagreement mask (흔들리는 픽셀만 1)
- a_final = a0 + β * m * r

기대 효과:
- Clean: disagreement 낮음 → m≈0 → a_final≈a0 (원본 보존)
- Corrupted: disagreement 높음 → m≈1 → 교정 적용

Based on SASv3 with:
- (A) No clamp for artifacts
- (B) MAD floor
- (1) Artifact scale normalization
- View-wise scalar fusion weights
"""

import copy
import math
from dataclasses import dataclass
from typing import Union, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SASv4Config:
    """SAS v4 Configuration with Residual Masking"""
    K: int = 5  # Number of views

    # Model type
    model: Literal["LGrad", "NPR"] = "LGrad"

    # Denoising target (artifact 추천)
    denoise_target: Literal["input", "artifact"] = "artifact"

    # Huber-TV 기본 파라미터
    huber_tv_lambda: float = 0.05
    huber_tv_delta: float = 0.01
    huber_tv_iterations: int = 3
    huber_tv_step_size: float = 0.2

    # View 생성 전략
    lambda_range: tuple[float, float] = (0.5, 1.5)
    jitter_strength: float = 0.1
    couple_delta_to_lambda: bool = True
    param_sets: Optional[list[tuple[float, float, int, float]]] = None

    # === Fusion 전략 ===
    fusion_scale: float = 2.0  # MAD 기반 가중치 민감도

    # === Residual + Disagreement Mask (SASv4 핵심!) ===
    use_residual_mask: bool = True  # Residual masking 활성화

    # Disagreement map 설정
    dmap_threshold: Optional[float] = None  # Auto calibration으로 설정 (p95)
    dmap_softness: float = 0.01  # Sigmoid softness (고정)
    dmap_percentile: float = 95.0  # Clean validation percentile

    # Residual blending
    residual_beta: float = 1.0  # Residual 강도 (0~1, 1=full correction)

    # Distribution matching (선택적, 기본 off)
    use_distribution_matching: bool = False

    device: str = "cuda"

    def __post_init__(self):
        """K개의 파라미터 세트 자동 생성"""
        if self.param_sets is None:
            self.param_sets = self._generate_param_sets()

        if len(self.param_sets) != self.K:
            raise ValueError(f"param_sets length must match K")

    def _generate_param_sets(self) -> list[tuple[float, float, int, float]]:
        """K개의 하이퍼파라미터 세트 생성"""
        import numpy as np

        param_sets = []
        base_lambda = self.huber_tv_lambda
        base_delta = self.huber_tv_delta

        # View 0: Original (no denoising)
        param_sets.append((0.0, 0.0, 0, 0.0))

        # View 1~K-1
        if self.K > 1:
            lambda_min = base_lambda * self.lambda_range[0]
            lambda_max = base_lambda * self.lambda_range[1]

            for i in range(self.K - 1):
                lambda_base = lambda_min + (lambda_max - lambda_min) * i / max(1, self.K - 2)

                jitter = np.random.uniform(-self.jitter_strength, self.jitter_strength)
                lambda_i = lambda_base * (1 + jitter)
                lambda_i = max(0.0, lambda_i)

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
    """Huber-TV denoising"""

    def __init__(self, config: SASv4Config):
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
        """Anisotropic Huber Total Variation denoising"""
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
            # Spatial gradients
            dx = torch.zeros_like(x_denoise)
            dx[:, :, :, :-1] = x_denoise[:, :, :, 1:] - x_denoise[:, :, :, :-1]

            dy = torch.zeros_like(x_denoise)
            dy[:, :, :-1, :] = x_denoise[:, :, 1:, :] - x_denoise[:, :, :-1, :]

            # Huber derivative
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

            # Gradient descent
            x_denoise = x_denoise - step_size * (
                (x_denoise - x) - lambda_tv * (div_x + div_y)
            )

            # Clamp only for input, not for artifacts
            if not skip_clamp:
                x_denoise = torch.clamp(x_denoise, 0, 1)

        return x_denoise


class UnifiedSASv4(nn.Module):
    """
    Unified SAS v4 for both LGrad and NPR

    핵심:
    - Residual-only correction with disagreement mask
    - 원본 artifact를 anchor로 유지
    - Multi-view disagreement가 큰 픽셀만 residual 적용
    - Clean 성능 보존 + Corrupted robustness 향상

    사용법:
        model = UnifiedSASv4(lgrad, config)
        model.calibrate_disagreement_threshold(clean_loader)  # 필수!
        logits = model(images)
    """

    def __init__(self, base_model, config: SASv4Config):
        super().__init__()
        self.cfg = config

        # Convert device to torch.device object for consistency
        self.device = torch.device(config.device)

        # Deep copy: move to CPU first to avoid device conflicts
        original_device = next(base_model.parameters()).device
        base_model_cpu = base_model.cpu()
        self.model = copy.deepcopy(base_model_cpu)

        # Move original back
        base_model.to(original_device)

        # Recursively ensure ALL modules and parameters are on target device
        self._move_to_device_recursive(self.model, self.device)

        # Update device attribute (convert to string for LGrad/NPR compatibility)
        if hasattr(self.model, 'device'):
            self.model.device = str(self.device)

        # For LGrad: explicitly ensure internal models are on correct device
        if hasattr(self.model, 'grad_model'):
            self._move_to_device_recursive(self.model.grad_model, self.device)
        if hasattr(self.model, 'classifier'):
            self._move_to_device_recursive(self.model.classifier, self.device)

        # Augmenter
        self.augmenter = StochasticAugmenter(config)

        # Validate
        if config.model not in ["LGrad", "NPR"]:
            raise ValueError(f"Unsupported model type: {config.model}")

        # Auto disagreement threshold (will be set by calibration)
        self.dmap_threshold_auto = None

        print(f"[SASv4] Initialized for {config.model} with K={config.K}")
        print(f"[SASv4] Denoise target: {config.denoise_target}")
        print(f"[SASv4] Fusion: Robust (view-wise scalar weight, scale={config.fusion_scale})")
        if config.use_residual_mask:
            print(f"[SASv4] Residual masking: ENABLED (β={config.residual_beta})")
            print(f"[SASv4] ⚠️  Call calibrate_disagreement_threshold(clean_loader) before use!")
        print(f"[SASv4] Parameter sets for {config.K} views:")
        for k, (lam, delta, iters, step) in enumerate(config.param_sets):
            print(f"  View {k}: λ={lam:.4f}, δ={delta:.4f}, iter={iters}, step={step:.2f}")

    def _move_to_device_recursive(self, module, device):
        """Recursively move all submodules and parameters to device"""
        module.to(device)

        # Explicitly move all children
        for child in module.children():
            self._move_to_device_recursive(child, device)

        # Explicitly move all parameters
        for param in module.parameters(recurse=False):
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(device)

        # Explicitly move all buffers
        for buffer_name, buffer in module.named_buffers(recurse=False):
            module._buffers[buffer_name] = buffer.to(device)

    def extract_multiview_artifacts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract K artifacts from multi-view

        Returns:
            artifacts_stack: [K, B, C, H, W]
            views_stack: [K, B, C, H, W]
        """
        B = x.shape[0]
        K = self.cfg.K

        all_artifacts = []
        all_views = []

        if self.cfg.denoise_target == "artifact":
            # Extract artifact once, denoise K times
            if self.cfg.model == "LGrad":
                artifact = self.model.img2grad(x)
            else:  # NPR
                artifact = self.model.img2npr(x)

            # Artifact scale normalization
            artifact_scale = artifact.abs().reshape(B, -1).median(dim=1)[0]
            artifact_scale = artifact_scale.view(B, 1, 1, 1) + 1e-8

            for k in range(K):
                lambda_tv, delta, iterations, step_size = self.cfg.param_sets[k]

                if k == 0:
                    artifact_k = artifact
                else:
                    # Normalize → TV → Denormalize
                    artifact_norm = artifact / artifact_scale
                    artifact_denoised_norm = self.augmenter.apply_anisotropic_huber_tv(
                        artifact_norm,
                        lambda_tv=lambda_tv,
                        huber_delta=delta,
                        iterations=iterations,
                        step_size=step_size,
                        skip_clamp=True
                    )
                    artifact_k = artifact_denoised_norm * artifact_scale

                all_artifacts.append(artifact_k)
                all_views.append(x)

        else:  # denoise_target == "input"
            for k in range(K):
                lambda_tv, delta, iterations, step_size = self.cfg.param_sets[k]

                if k == 0:
                    x_k = x
                else:
                    x_k = self.augmenter.apply_anisotropic_huber_tv(
                        x,
                        lambda_tv=lambda_tv,
                        huber_delta=delta,
                        iterations=iterations,
                        step_size=step_size
                    )

                if self.cfg.model == "LGrad":
                    artifact_k = self.model.img2grad(x_k)
                else:  # NPR
                    artifact_k = self.model.img2npr(x_k)

                all_artifacts.append(artifact_k)
                all_views.append(x_k)

        artifacts_stack = torch.stack(all_artifacts, dim=0)
        views_stack = torch.stack(all_views, dim=0)

        return artifacts_stack, views_stack

    def compute_robust_fusion_viewwise(self, artifacts_stack: torch.Tensor) -> torch.Tensor:
        """
        Robust fusion with view-wise scalar weight

        Args:
            artifacts_stack: [K, B, C, H, W]

        Returns:
            artifact_fused: [B, C, H, W]
        """
        K, B, C, H, W = artifacts_stack.shape
        eps = 1e-8

        # Median
        median_artifact = torch.median(artifacts_stack, dim=0)[0]  # [B, C, H, W]

        # Deviations
        deviations = torch.abs(artifacts_stack - median_artifact.unsqueeze(0))  # [K, B, C, H, W]

        # Per-view scalar: average over C, H, W
        d_k = deviations.mean(dim=(2, 3, 4))  # [K, B]

        # Global MAD with floor
        mad_global = torch.median(d_k, dim=0)[0]  # [B]
        mad_floor = 0.1 * mad_global.mean()
        mad_global = torch.maximum(mad_global, mad_floor)

        # Weights: [K, B]
        c = self.cfg.fusion_scale
        w = torch.exp(-d_k / (c * mad_global.unsqueeze(0)))
        w = w / (w.sum(dim=0, keepdim=True) + eps)  # Normalize

        # Weighted average: [K, B] → [K, B, 1, 1, 1]
        w_broadcast = w.view(K, B, 1, 1, 1)
        artifact_fused = (w_broadcast * artifacts_stack).sum(dim=0)

        return artifact_fused

    def fuse_artifacts(self, artifacts_stack: torch.Tensor) -> torch.Tensor:
        """
        Fuse artifacts using robust fusion with view-wise scalar weights

        Args:
            artifacts_stack: [K, B, C, H, W]

        Returns:
            artifact_fused: [B, C, H, W]
        """
        return self.compute_robust_fusion_viewwise(artifacts_stack)

    def compute_disagreement_map(self, artifacts_stack: torch.Tensor) -> torch.Tensor:
        """
        Compute pixel-wise disagreement map from multi-view artifacts

        핵심: 각 픽셀이 K개의 view에서 얼마나 흔들리는지 측정

        Args:
            artifacts_stack: [K, B, C, H, W]

        Returns:
            dmap: [B, 1, H, W] - 픽셀별 불일치 정도
        """
        # Step 1: Median artifact (robust center)
        median_artifact = torch.median(artifacts_stack, dim=0)[0]  # [B, C, H, W]

        # Step 2: Per-view deviation from median
        deviations = torch.abs(artifacts_stack - median_artifact.unsqueeze(0))  # [K, B, C, H, W]

        # Step 3: Average over K views and C channels → [B, H, W]
        # mean_{k, c} |a_k - median(a)|
        # dim=0: K (views), dim=2: C (channels)
        dmap = deviations.mean(dim=(0, 2))  # [B, H, W]

        # Step 4: Add channel dim for broadcasting
        dmap = dmap.unsqueeze(1)  # [B, 1, H, W]

        return dmap

    def compute_disagreement_mask(self, dmap: torch.Tensor) -> torch.Tensor:
        """
        Convert disagreement map to mask using sigmoid

        핵심: dmap이 threshold보다 높은 픽셀만 1에 가깝게

        Args:
            dmap: [B, 1, H, W]

        Returns:
            mask: [B, 1, H, W] in [0, 1]
        """
        # Use calibrated threshold if available
        if self.dmap_threshold_auto is not None:
            t = self.dmap_threshold_auto
        elif self.cfg.dmap_threshold is not None:
            t = self.cfg.dmap_threshold
        else:
            # Fallback: use median + z-score
            dmap_median = dmap.median()
            dmap_mad = (dmap - dmap_median).abs().median() + 1e-8
            t = dmap_median + 2.5 * dmap_mad

        s = self.cfg.dmap_softness

        # Sigmoid mask: m = sigmoid((d - t) / s)
        # d < t → m ≈ 0 (suppress)
        # d > t → m ≈ 1 (allow)
        mask = torch.sigmoid((dmap - t) / s)

        return mask

    def calibrate_disagreement_threshold(self, clean_validation_loader, percentile: float = None):
        """
        Calibrate disagreement map threshold using clean validation data

        핵심: Clean 이미지에서 disagreement의 p95를 threshold로 설정
        → Clean에서는 대부분 mask≈0, Corrupted에서만 mask≈1

        Args:
            clean_validation_loader: DataLoader with clean images
            percentile: Percentile to use (default: from config)
        """
        if percentile is None:
            percentile = self.cfg.dmap_percentile

        self.model.eval()
        all_dmaps = []

        print(f"[SASv4] Calibrating disagreement threshold on clean validation...")

        with torch.no_grad():
            for batch in clean_validation_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)

                # Extract multi-view artifacts
                artifacts_stack, _ = self.extract_multiview_artifacts(x)

                # Compute disagreement map
                dmap = self.compute_disagreement_map(artifacts_stack)  # [B, 1, H, W]

                # Move to CPU to save GPU memory
                all_dmaps.append(dmap.cpu())

        # Concatenate all dmaps
        all_dmaps = torch.cat(all_dmaps, dim=0)  # [N, 1, H, W]

        # 각 이미지의 대표값(95 percentile) 계산 (메모리 효율적)
        # 전체 픽셀을 quantile 하면 너무 커서 OOM 발생
        per_image_values = []
        for i in range(all_dmaps.shape[0]):
            dmap_i = all_dmaps[i]  # [1, H, W]
            # 각 이미지에서 상위 95% disagreement 값을 대표값으로 사용
            val = torch.quantile(dmap_i.flatten(), 0.95)
            per_image_values.append(val.item())

        per_image_values = torch.tensor(per_image_values)  # [N]

        # Clean 이미지들의 대표값 중 percentile 계산
        threshold = torch.quantile(per_image_values, percentile / 100.0)

        self.dmap_threshold_auto = threshold.item()
        print(f"[SASv4] Disagreement threshold calibrated: {self.dmap_threshold_auto:.6f} (p{percentile})")
        print(f"[SASv4] Clean images will have mask ≈ 0 (below threshold)")
        print(f"[SASv4] Corrupted images will have mask ≈ 1 (above threshold)")

    def forward(
        self,
        x: torch.Tensor,
        return_artifact: bool = False,
        return_masks: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass with Residual-only + Disagreement Mask

        핵심:
        - a0 = 원본 artifact (anchor)
        - af = fused artifact
        - r = af - a0 (residual)
        - m = disagreement mask
        - a_final = a0 + β * m * r

        Clean: m≈0 → a_final≈a0 (원본 보존)
        Corrupted: m≈1 → a_final≈af (교정 적용)
        """
        self.model.eval()

        # Step 1: Multi-view artifacts
        artifacts_stack, views_stack = self.extract_multiview_artifacts(x)

        # Step 2: Original artifact (anchor)
        a0 = artifacts_stack[0]  # [B, C, H, W]

        # Step 3: Fused artifact
        af = self.fuse_artifacts(artifacts_stack)  # [B, C, H, W]

        # Step 4: Compute residual
        r = af - a0  # [B, C, H, W]

        # Step 5: Compute disagreement map & mask
        if self.cfg.use_residual_mask:
            dmap = self.compute_disagreement_map(artifacts_stack)  # [B, 1, H, W]
            m = self.compute_disagreement_mask(dmap)  # [B, 1, H, W]

            # Step 6: Apply residual with mask
            # a_final = a0 + β * m * r
            beta = self.cfg.residual_beta
            a_final = a0 + beta * m * r  # [B, C, H, W]

        else:
            # Fallback: no masking (use fused artifact directly)
            a_final = af
            m = torch.ones_like(a0[:, :1, :, :])  # Dummy mask
            dmap = torch.zeros_like(a0[:, :1, :, :])

        # Step 7: (Optional) Distribution matching
        if self.cfg.use_distribution_matching:
            mean_orig = a0.mean(dim=(1, 2, 3), keepdim=True)
            std_orig = a0.std(dim=(1, 2, 3), keepdim=True)
            mean_final = a_final.mean(dim=(1, 2, 3), keepdim=True)
            std_final = a_final.std(dim=(1, 2, 3), keepdim=True)
            a_final = (a_final - mean_final) / (std_final + 1e-8) * std_orig + mean_orig

        # Step 8: Classify
        logits = self.model.classify(a_final)

        # Return
        returns = [logits]
        if return_artifact:
            returns.append(a_final)
        if return_masks:
            masks = {
                'artifact_final': a_final,
                'artifact_original': a0,
                'artifact_fused': af,
                'residual': r,
                'disagreement_map': dmap,
                'disagreement_mask': m,
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


# Utility functions for easy usage
def create_lgrad_sasv4(
    stylegan_weights: str,
    classifier_weights: str,
    K: int = 5,
    denoise_target: Literal["input", "artifact"] = "artifact",
    residual_beta: float = 1.0,
    device: str = "cuda",
) -> UnifiedSASv4:
    """
    Convenience function to create LGrad model with SASv4

    Args:
        stylegan_weights: Path to StyleGAN discriminator weights
        classifier_weights: Path to ResNet50 classifier weights
        K: Number of views
        denoise_target: Where to apply denoising ("input" or "artifact")
        residual_beta: Residual correction strength
        device: Device to use

    Returns:
        UnifiedSASv4 model ready for inference

    Example:
        >>> model = create_lgrad_sasv4(
        ...     stylegan_weights="weights/stylegan.pth",
        ...     classifier_weights="weights/classifier.pth",
        ...     K=5
        ... )
        >>> model.calibrate_disagreement_threshold(clean_loader)
        >>> probs = model.predict_proba(images)
    """
    from model.LGrad.lgrad_model import LGrad

    # Create base LGrad model
    lgrad = LGrad(
        stylegan_weights=stylegan_weights,
        classifier_weights=classifier_weights,
        device=device,
        resize=256,
    )

    # Apply SASv4
    config = SASv4Config(
        K=K,
        model="LGrad",
        denoise_target=denoise_target,
        residual_beta=residual_beta,
        device=device,
    )

    model = UnifiedSASv4(lgrad, config)

    return model


def create_npr_sasv4(
    weights: str,
    K: int = 5,
    denoise_target: Literal["input", "artifact"] = "artifact",
    residual_beta: float = 1.0,
    device: str = "cuda",
) -> UnifiedSASv4:
    """
    Convenience function to create NPR model with SASv4

    Args:
        weights: Path to NPR model weights
        K: Number of views
        denoise_target: Where to apply denoising ("input" or "artifact")
        residual_beta: Residual correction strength
        device: Device to use

    Returns:
        UnifiedSASv4 model ready for inference

    Example:
        >>> model = create_npr_sasv4(
        ...     weights="weights/npr.pth",
        ...     K=5
        ... )
        >>> model.calibrate_disagreement_threshold(clean_loader)
        >>> probs = model.predict_proba(images)
    """
    from model.NPR.npr_model import NPR

    # Create base NPR model
    npr = NPR(
        weights=weights,
        device=device,
    )

    # Apply SASv4
    config = SASv4Config(
        K=K,
        model="NPR",
        denoise_target=denoise_target,
        residual_beta=residual_beta,
        device=device,
    )

    model = UnifiedSASv4(npr, config)

    return model
