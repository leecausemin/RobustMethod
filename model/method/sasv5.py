"""
SAS v5 (Structure-Aware Sharpening v5) - GoG (Gradient of Gaussian) Approach

핵심 개선:
Huber-TV denoising → Gaussian blur로 교체

이유:
- Huber-TV는 edge-preserving이지만 Gaussian noise/JPEG block을 완전히 제거 못 함
- 결과적으로 모든 view가 비슷하게 망가져서 disagreement가 낮음
- Gaussian blur는 high-frequency noise를 직접 제거하면서 구조적 경계(artifact)는 보존

GoG (Gradient of Gaussian):
- view 생성: x_σ = GaussianBlur(x, σ)
- artifact: a_σ = img2grad(x_σ)
- σ = [0.0, 0.8, 1.6] → 다양한 smoothness → disagreement 생성

장점:
- Gaussian noise: blur로 직접 제거 ✅
- JPEG block: blur에서 약해짐 ✅
- Deepfake artifact (경계): 보존 ✅
- 하이퍼파라미터: σ 리스트 하나만 (Huber-TV는 4개)

Based on SASv4 with:
- Gaussian blur views (instead of Huber-TV)
- Residual + Disagreement mask (same as SASv4)
- View-wise scalar fusion weights (same as SASv4)
"""

import copy
import math
from dataclasses import dataclass
from typing import Union, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SASv5Config:
    """SAS v5 Configuration with GoG (Gradient of Gaussian)"""
    K: int = 3  # Number of views (σ 개수)

    # Model type
    model: Literal["LGrad", "NPR"] = "LGrad"

    # Denoising target (artifact 추천)
    denoise_target: Literal["input", "artifact"] = "artifact"

    # === GoG (Gradient of Gaussian) 파라미터 ===
    sigmas: Optional[list[float]] = None  # Gaussian blur σ 리스트
    # None이면 자동 생성: [0.0, 0.8, 1.6] (K=3 기준)

    # === Fusion 전략 ===
    fusion_scale: float = 2.0  # MAD 기반 가중치 민감도

    # === Residual + Disagreement Mask (SASv4와 동일) ===
    use_residual_mask: bool = True  # Residual masking 활성화

    # Disagreement map 설정
    dmap_threshold: Optional[float] = None  # Auto calibration
    dmap_softness: float = 0.01  # Sigmoid softness
    dmap_percentile: float = 95.0  # Clean validation percentile

    # Residual blending
    residual_beta: float = 1.0  # Residual 강도

    # Distribution matching (선택적)
    use_distribution_matching: bool = False

    device: str = "cuda"

    def __post_init__(self):
        """σ 리스트 자동 생성"""
        if self.sigmas is None:
            self.sigmas = self._generate_sigmas()

        if len(self.sigmas) != self.K:
            raise ValueError(f"sigmas length ({len(self.sigmas)}) must match K ({self.K})")

    def _generate_sigmas(self) -> list[float]:
        """
        K개의 σ 값 자동 생성

        전략:
        - σ=0: Original (no blur)
        - σ=0.8~1.6: Gaussian noise 제거하면서 구조 보존

        Returns:
            K개의 σ 값 리스트
        """
        if self.K == 1:
            return [0.0]
        elif self.K == 2:
            return [0.0, 1.2]
        elif self.K == 3:
            return [0.0, 0.8, 1.6]
        elif self.K == 5:
            return [0.0, 0.5, 1.0, 1.5, 2.0]
        else:
            # General case: 0부터 2.0까지 균등 간격
            import numpy as np
            sigmas = np.linspace(0.0, 2.0, self.K).tolist()
            return sigmas


class GaussianAugmenter:
    """Gaussian blur augmentation for GoG"""

    def __init__(self, config: SASv5Config):
        self.cfg = config

    def apply_gaussian_blur(
        self,
        x: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """
        Apply Gaussian blur to input

        Args:
            x: Input [B, C, H, W]
            sigma: Gaussian blur sigma

        Returns:
            x_blur: Blurred input [B, C, H, W]
        """
        if sigma == 0.0:
            # No blur
            return x

        B, C, H, W = x.shape

        # Create Gaussian kernel
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)  # 6σ rule
        kernel_size = max(3, kernel_size)  # Minimum size 3

        # 1D Gaussian
        x_coord = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
        x_coord = x_coord - kernel_size // 2
        gauss_1d = torch.exp(-x_coord**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()

        # 2D Gaussian (outer product)
        kernel = gauss_1d[:, None] * gauss_1d[None, :]
        kernel = kernel / kernel.sum()

        # Reshape for conv2d: [1, 1, K, K]
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        # Apply per-channel
        x_blur = torch.zeros_like(x)
        for c in range(C):
            x_c = x[:, c:c+1, :, :]  # [B, 1, H, W]
            x_c_blur = F.conv2d(
                x_c,
                kernel,
                padding=kernel_size // 2
            )
            x_blur[:, c:c+1, :, :] = x_c_blur

        return x_blur


class UnifiedSASv5(nn.Module):
    """
    Unified SAS v5 for both LGrad and NPR

    핵심:
    - GoG (Gradient of Gaussian) views
    - Residual + Disagreement mask (SASv4와 동일)
    - Gaussian/JPEG에 강함

    사용법:
        model = UnifiedSASv5(lgrad, config)
        model.calibrate_disagreement_threshold(clean_loader)  # 필수!
        logits = model(images)
    """

    def __init__(self, base_model, config: SASv5Config):
        super().__init__()
        self.cfg = config

        # Convert device to torch.device object
        self.device = torch.device(config.device)

        # Deep copy: move to CPU first
        original_device = next(base_model.parameters()).device
        base_model_cpu = base_model.cpu()
        self.model = copy.deepcopy(base_model_cpu)

        # Move original back
        base_model.to(original_device)

        # Move to target device
        self._move_to_device_recursive(self.model, self.device)

        # Update device attribute
        if hasattr(self.model, 'device'):
            self.model.device = str(self.device)

        # For LGrad: ensure internal models on correct device
        if hasattr(self.model, 'grad_model'):
            self._move_to_device_recursive(self.model.grad_model, self.device)
        if hasattr(self.model, 'classifier'):
            self._move_to_device_recursive(self.model.classifier, self.device)

        # Augmenter
        self.augmenter = GaussianAugmenter(config)

        # Validate
        if config.model not in ["LGrad", "NPR"]:
            raise ValueError(f"Unsupported model type: {config.model}")

        # Auto disagreement threshold (calibration으로 설정)
        self.dmap_threshold_auto = None

        print(f"[SASv5] Initialized for {config.model} with K={config.K}")
        print(f"[SASv5] GoG (Gradient of Gaussian) approach")
        print(f"[SASv5] Denoise target: {config.denoise_target}")
        print(f"[SASv5] Sigma values: {config.sigmas}")
        print(f"[SASv5] Fusion: Robust (view-wise scalar weight, scale={config.fusion_scale})")
        if config.use_residual_mask:
            print(f"[SASv5] Residual masking: ENABLED (β={config.residual_beta})")
            print(f"[SASv5] ⚠️  Call calibrate_disagreement_threshold(clean_loader) before use!")

    def _move_to_device_recursive(self, module, device):
        """Recursively move all submodules and parameters to device"""
        module.to(device)

        for child in module.children():
            self._move_to_device_recursive(child, device)

        for param in module.parameters(recurse=False):
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(device)

        for buffer_name, buffer in module.named_buffers(recurse=False):
            module._buffers[buffer_name] = buffer.to(device)

    def extract_multiview_artifacts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract K artifacts from K Gaussian-blurred views (GoG)

        핵심 변경점:
        - Huber-TV denoising → Gaussian blur
        - x_σ = GaussianBlur(x, σ)
        - a_σ = img2grad(x_σ) or img2npr(x_σ)

        Returns:
            artifacts_stack: [K, B, C, H, W]
            views_stack: [K, B, C, H, W]
        """
        B = x.shape[0]
        K = self.cfg.K

        all_artifacts = []
        all_views = []

        if self.cfg.denoise_target == "artifact":
            # Extract artifact once, then blur it K times
            if self.cfg.model == "LGrad":
                artifact = self.model.img2grad(x)
            else:  # NPR
                artifact = self.model.img2npr(x)

            # Apply Gaussian blur to artifact with K different σ
            for k in range(K):
                sigma = self.cfg.sigmas[k]

                # Apply Gaussian blur to artifact
                artifact_k = self.augmenter.apply_gaussian_blur(artifact, sigma)

                all_artifacts.append(artifact_k)
                all_views.append(x)

        else:  # denoise_target == "input"
            # Blur input K times, then extract artifact (GoG!)
            for k in range(K):
                sigma = self.cfg.sigmas[k]

                # Apply Gaussian blur to input
                x_k = self.augmenter.apply_gaussian_blur(x, sigma)

                # Extract artifact from blurred input
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
        Robust fusion with view-wise scalar weight (SASv4와 동일)

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

        # Weighted average
        w_broadcast = w.view(K, B, 1, 1, 1)
        artifact_fused = (w_broadcast * artifacts_stack).sum(dim=0)

        return artifact_fused

    def fuse_artifacts(self, artifacts_stack: torch.Tensor) -> torch.Tensor:
        """Fuse artifacts using robust fusion"""
        return self.compute_robust_fusion_viewwise(artifacts_stack)

    def compute_disagreement_map(self, artifacts_stack: torch.Tensor) -> torch.Tensor:
        """
        Compute pixel-wise disagreement map (SASv4와 동일)

        Args:
            artifacts_stack: [K, B, C, H, W]

        Returns:
            dmap: [B, 1, H, W]
        """
        # Median artifact
        median_artifact = torch.median(artifacts_stack, dim=0)[0]  # [B, C, H, W]

        # Per-view deviation
        deviations = torch.abs(artifacts_stack - median_artifact.unsqueeze(0))  # [K, B, C, H, W]

        # Average over K views and C channels → [B, H, W]
        dmap = deviations.mean(dim=(0, 2))  # [B, H, W]

        # Add channel dim
        dmap = dmap.unsqueeze(1)  # [B, 1, H, W]

        return dmap

    def compute_disagreement_mask(self, dmap: torch.Tensor) -> torch.Tensor:
        """
        Convert disagreement map to mask (SASv4와 동일)

        Args:
            dmap: [B, 1, H, W]

        Returns:
            mask: [B, 1, H, W]
        """
        # Use calibrated threshold if available
        if self.dmap_threshold_auto is not None:
            t = self.dmap_threshold_auto
        elif self.cfg.dmap_threshold is not None:
            t = self.cfg.dmap_threshold
        else:
            # Fallback
            dmap_median = dmap.median()
            dmap_mad = (dmap - dmap_median).abs().median() + 1e-8
            t = dmap_median + 2.5 * dmap_mad

        s = self.cfg.dmap_softness

        # Sigmoid mask
        mask = torch.sigmoid((dmap - t) / s)

        return mask

    def calibrate_disagreement_threshold(self, clean_validation_loader, percentile: float = None):
        """
        Calibrate disagreement map threshold (SASv4와 동일)

        Args:
            clean_validation_loader: DataLoader with clean images
            percentile: Percentile to use
        """
        if percentile is None:
            percentile = self.cfg.dmap_percentile

        self.model.eval()
        all_dmaps = []

        print(f"[SASv5] Calibrating disagreement threshold on clean validation...")

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

        # 각 이미지의 대표값(95 percentile) 계산
        per_image_values = []
        for i in range(all_dmaps.shape[0]):
            dmap_i = all_dmaps[i]  # [1, H, W]
            val = torch.quantile(dmap_i.flatten(), 0.95)
            per_image_values.append(val.item())

        per_image_values = torch.tensor(per_image_values)  # [N]

        # Clean 이미지들의 대표값 중 percentile 계산
        threshold = torch.quantile(per_image_values, percentile / 100.0)

        self.dmap_threshold_auto = threshold.item()
        print(f"[SASv5] Disagreement threshold calibrated: {self.dmap_threshold_auto:.6f} (p{percentile})")
        print(f"[SASv5] Clean images will have mask ≈ 0 (below threshold)")
        print(f"[SASv5] Corrupted images will have mask ≈ 1 (above threshold)")

    def forward(
        self,
        x: torch.Tensor,
        return_artifact: bool = False,
        return_masks: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass with GoG + Residual + Disagreement Mask

        핵심:
        - GoG views (Gaussian blur)
        - a_final = a0 + β * m * r (SASv4와 동일)
        """
        self.model.eval()

        # Step 1: Multi-view artifacts (GoG!)
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
            beta = self.cfg.residual_beta
            a_final = a0 + beta * m * r  # [B, C, H, W]

        else:
            # Fallback: no masking
            a_final = af
            m = torch.ones_like(a0[:, :1, :, :])
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


# Utility functions
def create_lgrad_sasv5(
    stylegan_weights: str,
    classifier_weights: str,
    K: int = 3,
    sigmas: Optional[list[float]] = None,
    denoise_target: Literal["input", "artifact"] = "artifact",
    residual_beta: float = 1.0,
    device: str = "cuda",
) -> UnifiedSASv5:
    """
    Convenience function to create LGrad model with SASv5 (GoG)

    Args:
        stylegan_weights: Path to StyleGAN weights
        classifier_weights: Path to classifier weights
        K: Number of views
        sigmas: Gaussian blur σ values (None for auto)
        denoise_target: "input" (recommended) or "artifact"
        residual_beta: Residual strength
        device: Device

    Returns:
        UnifiedSASv5 model

    Example:
        >>> model = create_lgrad_sasv5(
        ...     stylegan_weights="...",
        ...     classifier_weights="...",
        ...     K=3,
        ...     sigmas=[0.0, 0.8, 1.6]
        ... )
        >>> model.calibrate_disagreement_threshold(clean_loader)
        >>> probs = model.predict_proba(images)
    """
    from model.LGrad.lgrad_model import LGrad

    lgrad = LGrad(
        stylegan_weights=stylegan_weights,
        classifier_weights=classifier_weights,
        device=device,
        resize=256,
    )

    config = SASv5Config(
        K=K,
        model="LGrad",
        sigmas=sigmas,
        denoise_target=denoise_target,
        residual_beta=residual_beta,
        device=device,
    )

    model = UnifiedSASv5(lgrad, config)

    return model


def create_npr_sasv5(
    weights: str,
    K: int = 3,
    sigmas: Optional[list[float]] = None,
    denoise_target: Literal["input", "artifact"] = "artifact",
    residual_beta: float = 1.0,
    device: str = "cuda",
) -> UnifiedSASv5:
    """
    Convenience function to create NPR model with SASv5 (GoG)

    Args:
        weights: Path to NPR weights
        K: Number of views
        sigmas: Gaussian blur σ values (None for auto)
        denoise_target: "input" (recommended) or "artifact"
        residual_beta: Residual strength
        device: Device

    Returns:
        UnifiedSASv5 model

    Example:
        >>> model = create_npr_sasv5(
        ...     weights="...",
        ...     K=3,
        ...     sigmas=[0.0, 0.8, 1.6]
        ... )
        >>> model.calibrate_disagreement_threshold(clean_loader)
        >>> probs = model.predict_proba(images)
    """
    from model.NPR.npr_model import NPR

    npr = NPR(
        weights=weights,
        device=device,
    )

    config = SASv5Config(
        K=K,
        model="NPR",
        sigmas=sigmas,
        denoise_target=denoise_target,
        residual_beta=residual_beta,
        device=device,
    )

    model = UnifiedSASv5(npr, config)

    return model
