"""
SAS v3 (Structure-Aware Sharpening v3) - Simplified & Robust

핵심 개선:
1. Robust fusion with view-wise scalar weights (MAD 기반)
2. Artifact-level β blending (logit 말고 artifact에서 blend)
3. Auto β calibration (calibrate_beta() 호출 필수!)

Based on SAS with:
- (A) No clamp for artifacts
- (B) MAD floor
- (1) Artifact scale normalization
- (Bonus) Distribution matching (optional)

목표: Clean Acc 보존 + Corrupted Acc 향상 + 단순화
"""

import copy
import math
from dataclasses import dataclass
from typing import Union, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SASv3Config:
    """Simplified SAS v3 Configuration"""
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
    # Always use robust fusion with view-wise scalar weights
    fusion_scale: float = 2.0  # MAD 기반 가중치 민감도

    # === Adaptive β ===
    adaptive_beta: bool = True  # Corruption 자동 감지
    beta_method: Literal["auto", "manual"] = "auto"
    # - auto: Clean validation 95 percentile 기반 자동
    # - manual: 수동 threshold/softness 설정

    # Manual β 설정 (beta_method="manual"일 때만)
    beta_threshold: float = 0.02
    beta_softness: float = 0.01
    beta_min: float = 0.0
    beta_max: float = 1.0

    # Auto β 설정 (beta_method="auto"일 때만)
    beta_percentile: float = 95.0  # Clean validation disagreement의 percentile
    beta_auto_z: float = 2.5  # z-score threshold (fixed, 거의 튜닝 불필요)

    # Distribution matching (선택적)
    use_distribution_matching: bool = False  # Clean에서 분포 비틀 수 있어서 기본 off

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

    def __init__(self, config: SASv3Config):
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


class UnifiedSASv3(nn.Module):
    """
    Unified SAS v3 for both LGrad and NPR

    핵심:
    - Robust fusion with view-wise scalar weights (MAD 기반)
    - Artifact-level β blending (logit 말고)
    - Auto β calibration (calibrate_beta() 필수!)

    사용법:
        sas = UnifiedSASv3(lgrad, config)
        sas.calibrate_beta(clean_validation_loader)  # ← 필수!
        logits = sas(images)
    """

    def __init__(self, base_model, config: SASv3Config):
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

        # Auto β calibration placeholder
        self.beta_threshold_auto = None  # Will be set by calibrate_beta()

        print(f"[SASv3] Initialized for {config.model} with K={config.K}")
        print(f"[SASv3] Denoise target: {config.denoise_target}")
        print(f"[SASv3] Fusion: Robust (view-wise scalar weight, scale={config.fusion_scale})")
        if config.adaptive_beta:
            print(f"[SASv3] Adaptive β: {config.beta_method}")
            if config.beta_method == "auto":
                print(f"[SASv3] ⚠️  Call calibrate_beta(clean_loader) before use!")
        print(f"[SASv3] Parameter sets for {config.K} views:")
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
        Robust fusion with view-wise scalar weight (GPT 추천)

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

    def compute_disagreement(self, artifacts_stack: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-view disagreement for corruption detection

        Args:
            artifacts_stack: [K, B, C, H, W]

        Returns:
            disagreement: [B] per-sample disagreement score
        """
        median_artifact = torch.median(artifacts_stack, dim=0)[0]
        D = (artifacts_stack - median_artifact.unsqueeze(0)).abs().mean(dim=(0, 2, 3, 4))
        return D  # [B]

    def compute_adaptive_beta(self, disagreement: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive β from disagreement

        Args:
            disagreement: [B]

        Returns:
            beta: [B, 1, 1, 1]
        """
        if self.cfg.beta_method == "auto":
            # Auto calibration using z-score
            D_med = disagreement.median()
            D_mad = (disagreement - D_med).abs().median() + 1e-8

            # Use stored threshold if calibrated, otherwise use z-score
            if self.beta_threshold_auto is not None:
                t = self.beta_threshold_auto
                s = 1.0 * D_mad
            else:
                # Default: z-score based
                t = D_med + self.cfg.beta_auto_z * D_mad
                s = 1.0 * D_mad

            beta_raw = torch.sigmoid((disagreement - t) / s)

        else:  # manual
            t = self.cfg.beta_threshold
            s = self.cfg.beta_softness
            beta_raw = torch.sigmoid((disagreement - t) / s)

        # Clamp
        beta = self.cfg.beta_min + (self.cfg.beta_max - self.cfg.beta_min) * beta_raw

        return beta.view(-1, 1, 1, 1)  # [B, 1, 1, 1]

    def calibrate_beta(self, clean_validation_loader, percentile: float = None):
        """
        Calibrate β threshold using clean validation data

        Args:
            clean_validation_loader: DataLoader with clean images
            percentile: Percentile to use (default: from config)
        """
        if percentile is None:
            percentile = self.cfg.beta_percentile

        self.model.eval()
        disagreements = []

        with torch.no_grad():
            for batch in clean_validation_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)

                artifacts_stack, _ = self.extract_multiview_artifacts(x)
                D = self.compute_disagreement(artifacts_stack)
                disagreements.append(D)

        all_disagreements = torch.cat(disagreements, dim=0)
        threshold = torch.quantile(all_disagreements, percentile / 100.0)

        self.beta_threshold_auto = threshold.item()
        print(f"[SASv3] Auto β calibrated: threshold={self.beta_threshold_auto:.6f} (p{percentile})")

    def forward(
        self,
        x: torch.Tensor,
        return_artifact: bool = False,
        return_masks: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass with SAS v3

        핵심: Artifact-level β blending (not logit-level)
        """
        self.model.eval()

        # Step 1: Multi-view artifacts
        artifacts_stack, views_stack = self.extract_multiview_artifacts(x)

        # Step 2: Fuse artifacts
        artifact_enhanced = self.fuse_artifacts(artifacts_stack)  # [B, C, H, W]

        # Step 3: (Optional) Distribution matching
        if self.cfg.use_distribution_matching:
            artifact_original = artifacts_stack[0]
            mean_orig = artifact_original.mean(dim=(1, 2, 3), keepdim=True)
            std_orig = artifact_original.std(dim=(1, 2, 3), keepdim=True)
            mean_enh = artifact_enhanced.mean(dim=(1, 2, 3), keepdim=True)
            std_enh = artifact_enhanced.std(dim=(1, 2, 3), keepdim=True)
            artifact_enhanced = (artifact_enhanced - mean_enh) / (std_enh + 1e-8) * std_orig + mean_orig

        # Step 4: Adaptive β (artifact-level blending)
        if self.cfg.adaptive_beta:
            disagreement = self.compute_disagreement(artifacts_stack)  # [B]
            beta = self.compute_adaptive_beta(disagreement)  # [B, 1, 1, 1]

            artifact_original = artifacts_stack[0]  # View 0
            artifact_final = (1 - beta) * artifact_original + beta * artifact_enhanced
        else:
            artifact_final = artifact_enhanced
            beta = torch.ones(x.shape[0], 1, 1, 1, device=x.device)
            disagreement = torch.zeros(x.shape[0], device=x.device)

        # Step 5: Classify
        logits = self.model.classify(artifact_final)

        # Return
        returns = [logits]
        if return_artifact:
            returns.append(artifact_final)
        if return_masks:
            masks = {
                'artifact_final': artifact_final,
                'artifact_enhanced': artifact_enhanced,
                'artifact_original': artifacts_stack[0],
                'beta': beta,
                'disagreement': disagreement,
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
