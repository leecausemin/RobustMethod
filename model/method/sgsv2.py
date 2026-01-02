"""
SGS v2 (Stochastic Gradient Smoothing v2) with Adaptive Application

핵심 개선:
1. Multi-view disagreement 기반 자동 corruption 감지
2. Clean image에서는 거의 identity (β≈0)
3. Corrupted image에서만 강하게 적용 (β≈1)
4. Logit-level blending으로 원본 성능 보존

Based on SAS with all improvements:
- (A) No clamp for artifacts
- (B) MAD floor
- (C) Mean + Robust hybrid
- (1) Artifact scale normalization
- (Bonus) Distribution matching
"""

import copy
import math
from dataclasses import dataclass
from typing import Union, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SGSv2Config:
    """Configuration for SGS v2 (Adaptive SAS)"""
    K: int = 5  # Number of views to sample

    # Model type
    model: Literal["LGrad", "NPR"] = "LGrad"

    # Huber-TV 적용 대상 (artifact 추천)
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

    # Robust fusion 설정
    fusion_scale: float = 2.0  # c in exp(-d_k / (c * mad)), 1~3 추천
    fusion_alpha: float = 0.3  # Robust 비중 (0.2~0.4 추천)

    # === SGS v2 전용: Adaptive Application ===
    adaptive_beta: bool = True  # β 자동 조절 활성화
    beta_threshold: float = 0.02  # Disagreement threshold (t)
    beta_softness: float = 0.01  # Sigmoid softness (s)
    beta_min: float = 0.0  # Minimum β (완전 original)
    beta_max: float = 1.0  # Maximum β (완전 SAS)

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

    def __init__(self, config: SGSv2Config):
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


class UnifiedSGSv2(nn.Module):
    """
    Unified SGS v2 (Adaptive Stochastic Gradient Smoothing) for both LGrad and NPR

    핵심 개선:
    - Multi-view disagreement로 corruption 자동 감지
    - Clean → β≈0 (원본 유지)
    - Corrupted → β≈1 (SAS 강하게 적용)
    - Logit-level blending으로 원본 성능 보존
    """

    def __init__(self, base_model, config: SGSv2Config):
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

        print(f"[SGSv2] Initialized for {config.model} with K={config.K}, denoise_target={config.denoise_target}")
        print(f"[SGSv2] Base params: λ={config.huber_tv_lambda}, δ={config.huber_tv_delta}")
        print(f"[SGSv2] Fusion: scale={config.fusion_scale}, alpha={config.fusion_alpha}")
        if config.adaptive_beta:
            print(f"[SGSv2] Adaptive β: threshold={config.beta_threshold}, softness={config.beta_softness}")
        print(f"[SGSv2] Parameter sets for {config.K} views:")
        for k, (lam, delta, iters, step) in enumerate(config.param_sets):
            print(f"  View {k}: λ={lam:.4f}, δ={delta:.4f}, iter={iters}, step={step:.2f}")

    def extract_multiview_artifacts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract K artifacts from K denoised views

        Returns:
            artifacts_stack: [K, B, C, H, W] K개의 artifact
            views_stack: [K, B, C, H, W] K개의 view images
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
            artifact_scale = artifact.abs().reshape(B, -1).median(dim=1)[0]
            artifact_scale = artifact_scale.view(B, 1, 1, 1) + 1e-8

            for k in range(K):
                lambda_tv, delta, iterations, step_size = self.cfg.param_sets[k]

                if k == 0:
                    # View 0: original (no denoising)
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
                        skip_clamp=True  # artifact는 clamp 안 함
                    )
                    artifact_k = artifact_denoised_norm * artifact_scale

                all_artifacts.append(artifact_k)
                all_views.append(x)

        else:  # denoise_target == "input"
            # Denoise input K times, then extract artifact
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

                # Extract artifact
                if self.cfg.model == "LGrad":
                    artifact_k = self.model.img2grad(x_k)
                else:  # NPR
                    artifact_k = self.model.img2npr(x_k)

                all_artifacts.append(artifact_k)
                all_views.append(x_k)

        # Stack: [K, B, C, H, W]
        artifacts_stack = torch.stack(all_artifacts, dim=0)
        views_stack = torch.stack(all_views, dim=0)

        return artifacts_stack, views_stack

    def compute_robust_fusion(
        self,
        artifacts_stack: torch.Tensor
    ) -> torch.Tensor:
        """
        Robust fusion with MAD floor (GPT 패치 B)
        """
        K, B, C, H, W = artifacts_stack.shape
        eps = 1e-8

        # Median artifact
        median_artifact = torch.median(artifacts_stack, dim=0)[0]

        # Deviations
        deviations = torch.abs(artifacts_stack - median_artifact.unsqueeze(0))

        # MAD
        mad = torch.median(deviations, dim=0)[0]

        # GPT 패치 (B): MAD floor
        global_mad = mad.mean()
        mad_floor = 0.1 * global_mad
        mad = torch.maximum(mad, mad_floor)

        # Weights
        c = self.cfg.fusion_scale
        weights = torch.exp(-deviations / (c * mad.unsqueeze(0)))

        # Weighted average
        weighted_sum = (weights * artifacts_stack).sum(dim=0)
        weight_sum = weights.sum(dim=0)
        artifact_fused = weighted_sum / (weight_sum + eps)

        return artifact_fused

    def compute_disagreement(self, artifacts_stack: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-view disagreement score (corruption detection)

        Args:
            artifacts_stack: [K, B, C, H, W]

        Returns:
            disagreement: [B, 1, 1, 1] - per-sample disagreement score
        """
        # Median artifact
        median_artifact = torch.median(artifacts_stack, dim=0)[0]  # [B, C, H, W]

        # Per-sample disagreement: mean absolute deviation from median
        # Average over K views, C channels, H, W → [B]
        # artifacts_stack: [K, B, C, H, W], median: [B, C, H, W]
        # deviation: [K, B, C, H, W], mean over (0, 2, 3, 4) → [B]
        D = (artifacts_stack - median_artifact.unsqueeze(0)).abs().mean(dim=(0, 2, 3, 4))

        # Reshape to [B, 1, 1, 1] for broadcasting
        D = D.view(-1, 1, 1, 1)

        return D

    def compute_adaptive_beta(self, disagreement: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive β from disagreement score

        Args:
            disagreement: [B, 1, 1, 1]

        Returns:
            beta: [B, 1, 1, 1] in [beta_min, beta_max]
        """
        t = self.cfg.beta_threshold
        s = self.cfg.beta_softness

        # β = sigmoid((D - t) / s)
        beta_raw = torch.sigmoid((disagreement - t) / s)

        # Clamp to [beta_min, beta_max]
        beta = self.cfg.beta_min + (self.cfg.beta_max - self.cfg.beta_min) * beta_raw

        return beta

    def forward(
        self,
        x: torch.Tensor,
        return_artifact: bool = False,
        return_masks: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass with Adaptive SGS v2

        핵심: Logit-level blending with adaptive β
        - Clean: β≈0 → logits ≈ logit_original
        - Corrupted: β≈1 → logits ≈ logit_SAS
        """
        self.model.eval()

        # Step 1: Extract K artifacts from multi-view
        artifacts_stack, views_stack = self.extract_multiview_artifacts(x)

        # Step 2: Compute both mean and robust fusion
        artifact_mean = artifacts_stack.mean(dim=0)
        artifact_robust = self.compute_robust_fusion(artifacts_stack)

        # GPT 패치 (C): Mean + Robust hybrid
        alpha = self.cfg.fusion_alpha
        artifact_enhanced = (1 - alpha) * artifact_mean + alpha * artifact_robust

        # Bonus: Distribution matching
        artifact_original = artifacts_stack[0]
        mean_orig = artifact_original.mean(dim=(1, 2, 3), keepdim=True)
        std_orig = artifact_original.std(dim=(1, 2, 3), keepdim=True)
        mean_fused = artifact_enhanced.mean(dim=(1, 2, 3), keepdim=True)
        std_fused = artifact_enhanced.std(dim=(1, 2, 3), keepdim=True)
        artifact_enhanced = (artifact_enhanced - mean_fused) / (std_fused + 1e-8) * std_orig + mean_orig

        # Step 3: Adaptive β based on disagreement (SGSv2 핵심!)
        if self.cfg.adaptive_beta:
            disagreement = self.compute_disagreement(artifacts_stack)  # [B, 1, 1, 1]
            beta = self.compute_adaptive_beta(disagreement)  # [B, 1, 1, 1]

            # Classify both original and enhanced
            logit_original = self.model.classify(artifact_original)  # [B, 1]
            logit_enhanced = self.model.classify(artifact_enhanced)  # [B, 1]

            # Logit-level blending (β를 [B, 1]로 squeeze)
            beta_scalar = beta.squeeze(-1).squeeze(-1)  # [B, 1]
            logits = (1 - beta_scalar) * logit_original + beta_scalar * logit_enhanced
        else:
            # No adaptive β: just use enhanced artifact
            logits = self.model.classify(artifact_enhanced)
            beta = torch.ones(x.shape[0], 1, 1, 1, device=x.device)
            disagreement = torch.zeros(x.shape[0], 1, 1, 1, device=x.device)

        # Return based on flags
        returns = [logits]
        if return_artifact:
            returns.append(artifact_enhanced)
        if return_masks:
            masks = {
                'artifact_fused': artifact_enhanced,
                'artifact_mean': artifact_mean,
                'artifact_robust': artifact_robust,
                'artifact_original': artifact_original,
                'beta': beta,  # Adaptive strength
                'disagreement': disagreement,  # Corruption score
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
