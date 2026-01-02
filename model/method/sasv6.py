"""
SAS v6 (Structure-Aware Sharpening v6) - Gaussian + Block DCT + DWT Preprocessing

핵심 아이디어:
LGrad/NPR 계열 모델 앞에 붙이는 전처리 파이프라인으로,
Gaussian + JPEG 노이즈에 강건한 아티팩트 추출을 위해:

파이프라인:
I -> Gaussian blur -> Block DCT/IDCT -> DWT -> threshold -> iDWT

1. 약한 Gaussian smoothing: 순수 노이즈 제거, 구조 유지
2. Block DCT 기반 JPEG 정규화: JPEG 품질 차이 통일
3. Residual 아티팩트 채널 생성: R_g = I - I_g, R_d = I_g - I_d
4. DWT 대역 분해 + threshold: 고주파 서브밴드 soft-threshold
5. iDWT 복원: 정제된 이미지 생성

Based on CNN·DWT 동작 원리 및 JPEG 압축 특성
"""

import copy
from dataclasses import dataclass
from typing import Union, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


@dataclass
class SASv6Config:
    """SAS v6 Configuration - Gaussian + DCT + DWT Preprocessing"""

    # Model type
    model: Literal["LGrad", "NPR"] = "LGrad"

    # === Gaussian Smoothing ===
    gaussian_kernel_size: int = 3  # Gaussian kernel size (3 or 5)
    gaussian_sigma: float = 0.8  # Gaussian sigma (0.5~1.0)

    # === Block DCT Normalization ===
    dct_block_size: int = 8  # DCT block size (8x8 standard)
    dct_threshold: float = 2.5  # Soft-threshold for DCT coefficients
    dct_clip: float = 10.0  # Clipping value for DCT coefficients

    # === DWT Parameters ===
    wavelet_type: str = 'haar'  # Wavelet type ('haar', 'db2', 'db4', etc.)
    dwt_threshold: float = 0.05  # Soft-threshold for high-freq subbands (λ)

    # === Residual Configuration ===
    use_residuals: bool = True  # Whether to use residual channels
    residual_type: Literal["R_g", "R_d", "both"] = "R_d"  # Which residual to use
    # R_g = I - I_g (noise/fine artifacts)
    # R_d = I_g - I_d (structural JPEG/synthesis differences)

    # === Input Construction ===
    input_mode: Literal["I_d_wave", "concat", "residual_only"] = "concat"
    # "I_d_wave": Only use wavelet-refined I_d
    # "concat": [I_d_wave, R_d_wave] concatenated
    # "residual_only": Only use R_d_wave

    # === Advanced Options ===
    apply_dwt_to_input: bool = True  # Apply DWT to I_d
    apply_dwt_to_residual: bool = True  # Apply DWT to R_d

    device: str = "cuda"

    def __post_init__(self):
        """Validate configuration"""
        if self.gaussian_kernel_size % 2 == 0:
            raise ValueError("gaussian_kernel_size must be odd")

        if self.wavelet_type not in pywt.wavelist():
            raise ValueError(f"Invalid wavelet type: {self.wavelet_type}")


class SASv6Preprocessor:
    """
    SASv6 전처리 파이프라인

    Gaussian → Block DCT → DWT → threshold → iDWT
    """

    def __init__(self, config: SASv6Config):
        self.cfg = config

    # ---------------------------
    # 1. Gaussian Smoothing
    # ---------------------------

    def gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply weak Gaussian smoothing

        Args:
            x: (B, C, H, W)

        Returns:
            x_blurred: (B, C, H, W)
        """
        kernel_size = self.cfg.gaussian_kernel_size
        sigma = self.cfg.gaussian_sigma

        # Create Gaussian kernel
        coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - (kernel_size - 1) / 2.
        grid = coords[None, :]**2 + coords[:, None]**2
        kernel = torch.exp(-grid / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(x.shape[1], 1, 1, 1)  # depthwise

        # Apply convolution with reflection padding
        x_pad = F.pad(x, [kernel_size//2]*4, mode='reflect')
        out = F.conv2d(x_pad, kernel, groups=x.shape[1])

        return out

    # ---------------------------
    # 2. Block DCT Normalization
    # ---------------------------

    def dct_2d(self, block: torch.Tensor) -> torch.Tensor:
        """
        2D DCT using FFT

        Args:
            block: (..., 8, 8)

        Returns:
            coeff: (..., 8, 8)
        """
        # Use rfft2 for real inputs
        return torch.fft.fft2(block, norm='ortho').real

    def idct_2d(self, coeff: torch.Tensor) -> torch.Tensor:
        """
        2D inverse DCT using IFFT

        Args:
            coeff: (..., 8, 8)

        Returns:
            block: (..., 8, 8)
        """
        # Convert to complex for ifft2
        coeff_complex = coeff.to(torch.complex64)
        return torch.fft.ifft2(coeff_complex, norm='ortho').real

    def block_dct_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply block DCT normalization (JPEG regularization)

        핵심: 8×8 블록별로 DCT → soft-threshold & clipping → IDCT
        목적: JPEG 품질 차이에 따른 고주파/블록 스케일 차이 통일

        Args:
            x: (B, C, H, W) - assumed H, W are multiples of 8

        Returns:
            x_normalized: (B, C, H, W)
        """
        B, C, H, W = x.shape
        block_size = self.cfg.dct_block_size

        # Pad to multiples of block_size if needed
        pad_h = (block_size - H % block_size) % block_size
        pad_w = (block_size - W % block_size) % block_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
            H_pad, W_pad = x.shape[2], x.shape[3]
        else:
            H_pad, W_pad = H, W

        # Unfold into blocks: (B, C, H//8, W//8, 8, 8)
        patches = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        Bc, Cc, Ph, Pw, _, _ = patches.shape
        patches = patches.contiguous().view(Bc, Cc, Ph, Pw, block_size, block_size)

        # DCT
        coeff = self.dct_2d(patches)

        # Soft-threshold & clipping
        thresh = self.cfg.dct_threshold
        clip = self.cfg.dct_clip

        coeff_abs = coeff.abs()
        # Soft threshold: reduce small coefficients
        mask_small = coeff_abs < thresh
        coeff = torch.where(mask_small, coeff * (coeff_abs / (thresh + 1e-8)), coeff)
        # Clip large coefficients
        coeff = coeff.clamp(-clip, clip)

        # IDCT
        rec_blocks = self.idct_2d(coeff)

        # Fold back: (B, C, Ph, Pw, 8, 8) -> (B, C, H_pad, W_pad)
        rec_blocks = rec_blocks.view(B, C, Ph, Pw, block_size, block_size)
        rec_blocks = rec_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        rec_blocks = rec_blocks.view(B, C, Ph * block_size, Pw * block_size)

        # Crop to original size
        if pad_h > 0 or pad_w > 0:
            rec_blocks = rec_blocks[:, :, :H, :W]

        return rec_blocks

    # ---------------------------
    # 3. Residual Maps
    # ---------------------------

    def make_residuals(self, I: torch.Tensor) -> tuple:
        """
        Generate residual maps

        Args:
            I: (B, C, H, W) - input image

        Returns:
            I_g: Gaussian blurred
            I_d: DCT normalized
            R_g: I - I_g (noise/fine artifacts)
            R_d: I_g - I_d (structural differences)
        """
        I_g = self.gaussian_blur(I)
        I_d = self.block_dct_normalize(I_g)
        R_g = I - I_g
        R_d = I_g - I_d

        return I_g, I_d, R_g, R_d

    # ---------------------------
    # 4. DWT + Threshold + iDWT
    # ---------------------------

    def dwt2_batch(self, x: torch.Tensor) -> tuple:
        """
        Batch 2D DWT using pywt

        Args:
            x: (B, C, H, W)

        Returns:
            LL, LH, HL, HH: each (B, C, H//2, W//2)
        """
        B, C, H, W = x.shape
        wave = self.cfg.wavelet_type

        LL_list, LH_list, HL_list, HH_list = [], [], [], []

        for b in range(B):
            LL_c, LH_c, HL_c, HH_c = [], [], [], []
            for c in range(C):
                arr = x[b, c].detach().cpu().numpy()
                coeffs2 = pywt.dwt2(arr, wave)
                LL, (LH, HL, HH) = coeffs2
                LL_c.append(torch.tensor(LL, dtype=x.dtype))
                LH_c.append(torch.tensor(LH, dtype=x.dtype))
                HL_c.append(torch.tensor(HL, dtype=x.dtype))
                HH_c.append(torch.tensor(HH, dtype=x.dtype))
            LL_list.append(torch.stack(LL_c, dim=0))
            LH_list.append(torch.stack(LH_c, dim=0))
            HL_list.append(torch.stack(HL_c, dim=0))
            HH_list.append(torch.stack(HH_c, dim=0))

        LL = torch.stack(LL_list, dim=0).to(x.device)
        LH = torch.stack(LH_list, dim=0).to(x.device)
        HL = torch.stack(HL_list, dim=0).to(x.device)
        HH = torch.stack(HH_list, dim=0).to(x.device)

        return LL, LH, HL, HH

    def idwt2_batch(self, LL: torch.Tensor, LH: torch.Tensor,
                    HL: torch.Tensor, HH: torch.Tensor) -> torch.Tensor:
        """
        Batch 2D inverse DWT using pywt

        Args:
            LL, LH, HL, HH: each (B, C, H//2, W//2)

        Returns:
            x: (B, C, H, W)
        """
        B, C, H, W = LL.shape
        wave = self.cfg.wavelet_type

        out = []
        for b in range(B):
            rec_c = []
            for c in range(C):
                coeffs2 = (
                    LL[b, c].detach().cpu().numpy(),
                    (
                        LH[b, c].detach().cpu().numpy(),
                        HL[b, c].detach().cpu().numpy(),
                        HH[b, c].detach().cpu().numpy(),
                    )
                )
                rec = pywt.idwt2(coeffs2, wave)
                rec_c.append(torch.tensor(rec, dtype=LL.dtype))
            out.append(torch.stack(rec_c, dim=0))

        out = torch.stack(out, dim=0).to(LL.device)
        return out

    def soft_threshold(self, x: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Soft-thresholding operator

        Args:
            x: input tensor
            lam: threshold value

        Returns:
            thresholded tensor
        """
        mag = x.abs()
        return torch.sign(x) * torch.clamp(mag - lam, min=0.0)

    def wavelet_threshold(self, LL: torch.Tensor, LH: torch.Tensor,
                         HL: torch.Tensor, HH: torch.Tensor) -> tuple:
        """
        Apply soft-threshold to high-frequency subbands

        Args:
            LL, LH, HL, HH: DWT subbands

        Returns:
            LL, LH_t, HL_t, HH_t: thresholded subbands
        """
        lam = self.cfg.dwt_threshold

        # Only threshold high-frequency bands
        LH_t = self.soft_threshold(LH, lam)
        HL_t = self.soft_threshold(HL, lam)
        HH_t = self.soft_threshold(HH, lam)

        return LL, LH_t, HL_t, HH_t

    def make_wavelet_refined(self, I_d: torch.Tensor, R_d: torch.Tensor) -> tuple:
        """
        Apply DWT + threshold + iDWT to refine images

        Args:
            I_d: DCT normalized image (B, C, H, W)
            R_d: Residual (B, C, H, W)

        Returns:
            I_d_wave: wavelet-refined I_d
            R_d_wave: wavelet-refined R_d
        """
        # Process I_d
        if self.cfg.apply_dwt_to_input:
            LLd, LHd, HLd, HHd = self.dwt2_batch(I_d)
            LLd, LHd, HLd, HHd = self.wavelet_threshold(LLd, LHd, HLd, HHd)
            I_d_wave = self.idwt2_batch(LLd, LHd, HLd, HHd)
        else:
            I_d_wave = I_d

        # Process R_d
        if self.cfg.apply_dwt_to_residual:
            LLrd, LHrd, HLrd, HHrd = self.dwt2_batch(R_d)
            LLrd, LHrd, HLrd, HHrd = self.wavelet_threshold(LLrd, LHrd, HLrd, HHrd)
            R_d_wave = self.idwt2_batch(LLrd, LHrd, HLrd, HHrd)
        else:
            R_d_wave = R_d

        return I_d_wave, R_d_wave

    def preprocess(self, I: torch.Tensor) -> torch.Tensor:
        """
        Full preprocessing pipeline

        Args:
            I: (B, C, H, W) - input image

        Returns:
            preprocessed: (B, C', H, W) - preprocessed input for LGrad/NPR
        """
        # Step 1: Generate residuals
        I_g, I_d, R_g, R_d = self.make_residuals(I)

        # Step 2: Apply wavelet refinement
        I_d_wave, R_d_wave = self.make_wavelet_refined(I_d, R_d)

        # Step 3: Construct input based on config
        if self.cfg.input_mode == "I_d_wave":
            # Only use wavelet-refined I_d
            return I_d_wave

        elif self.cfg.input_mode == "concat":
            # Concatenate [I_d_wave, R_d_wave]
            if self.cfg.use_residuals:
                if self.cfg.residual_type == "R_d":
                    return torch.cat([I_d_wave, R_d_wave], dim=1)
                elif self.cfg.residual_type == "R_g":
                    _, R_g_wave = self.make_wavelet_refined(I_d, R_g)
                    return torch.cat([I_d_wave, R_g_wave], dim=1)
                elif self.cfg.residual_type == "both":
                    _, R_g_wave = self.make_wavelet_refined(I_d, R_g)
                    return torch.cat([I_d_wave, R_g_wave, R_d_wave], dim=1)
            else:
                return I_d_wave

        elif self.cfg.input_mode == "residual_only":
            # Only use residual
            return R_d_wave

        else:
            raise ValueError(f"Unknown input_mode: {self.cfg.input_mode}")


class UnifiedSASv6(nn.Module):
    """
    Unified SASv6 for both LGrad and NPR

    핵심:
    - LGrad/NPR 모델 앞에 Gaussian + DCT + DWT 전처리 파이프라인 적용
    - Gaussian + JPEG 노이즈에 강건한 아티팩트 추출
    - 기존 모델의 학습 가중치는 그대로 사용

    사용법:
        config = SASv6Config(model="LGrad", input_mode="concat")
        model = UnifiedSASv6(lgrad, config)
        logits = model(images)
    """

    def __init__(self, base_model, config: SASv6Config):
        super().__init__()
        self.cfg = config

        # Convert device to torch.device object
        self.device = torch.device(config.device)

        # Deep copy base model
        original_device = next(base_model.parameters()).device
        base_model_cpu = base_model.cpu()
        self.model = copy.deepcopy(base_model_cpu)

        # Move original back
        base_model.to(original_device)

        # Move model to target device
        self._move_to_device_recursive(self.model, self.device)

        # Update device attribute
        if hasattr(self.model, 'device'):
            self.model.device = str(self.device)

        # For LGrad: ensure internal models are on correct device
        if hasattr(self.model, 'grad_model'):
            self._move_to_device_recursive(self.model.grad_model, self.device)
        if hasattr(self.model, 'classifier'):
            self._move_to_device_recursive(self.model.classifier, self.device)

        # Preprocessor
        self.preprocessor = SASv6Preprocessor(config)

        # Validate
        if config.model not in ["LGrad", "NPR"]:
            raise ValueError(f"Unsupported model type: {config.model}")

        # Determine expected input channels for the model
        self._setup_input_channels()

        print(f"[SASv6] Initialized for {config.model}")
        print(f"[SASv6] Gaussian: kernel={config.gaussian_kernel_size}, σ={config.gaussian_sigma}")
        print(f"[SASv6] Block DCT: thresh={config.dct_threshold}, clip={config.dct_clip}")
        print(f"[SASv6] DWT: type={config.wavelet_type}, λ={config.dwt_threshold}")
        print(f"[SASv6] Input mode: {config.input_mode}")
        if config.use_residuals:
            print(f"[SASv6] Residual type: {config.residual_type}")

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

    def _setup_input_channels(self):
        """
        Setup input channel handling

        문제: preprocessor가 concat 모드일 때 채널이 6개가 될 수 있는데,
        기존 LGrad/NPR는 3채널만 받음

        해결:
        - LGrad: img2grad를 통과하면 gradient가 되므로,
          concatenated input을 그냥 넘겨도 gradient 추출 시 처리됨
        - NPR: 마찬가지로 img2npr가 알아서 처리

        여기서는 입력 채널 수가 달라지는 경우를 체크만 함
        """
        if self.cfg.input_mode == "concat" and self.cfg.use_residuals:
            if self.cfg.residual_type == "both":
                self.expected_channels = 9  # I_d_wave(3) + R_g_wave(3) + R_d_wave(3)
            else:
                self.expected_channels = 6  # I_d_wave(3) + residual(3)
        else:
            self.expected_channels = 3  # Normal 3-channel input

        # For LGrad/NPR, we need to handle this in forward pass
        # by either:
        # 1. Passing concatenated channels through img2grad/img2npr
        # 2. Processing each 3-channel separately and combining artifacts

        # We'll use option 2: process separately and combine
        self.use_channel_split = self.expected_channels > 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SASv6 preprocessing

        Args:
            x: (B, 3, H, W) - input image [0, 1]

        Returns:
            logits: (B, 1) - classification logits
        """
        self.model.eval()

        # Step 1: Preprocess
        x_preprocessed = self.preprocessor.preprocess(x)  # (B, C', H, W)

        # Step 2: Extract artifact and classify
        if self.use_channel_split:
            # Split channels and process separately
            B, C, H, W = x_preprocessed.shape
            n_groups = C // 3

            artifacts = []
            for i in range(n_groups):
                x_i = x_preprocessed[:, i*3:(i+1)*3, :, :]  # (B, 3, H, W)

                if self.cfg.model == "LGrad":
                    artifact_i = self.model.img2grad(x_i)
                else:  # NPR
                    artifact_i = self.model.img2npr(x_i)

                artifacts.append(artifact_i)

            # Combine artifacts (average)
            artifact_combined = torch.stack(artifacts, dim=0).mean(dim=0)

            # Classify
            logits = self.model.classify(artifact_combined)

        else:
            # Direct processing
            if self.cfg.model == "LGrad":
                artifact = self.model.img2grad(x_preprocessed)
            else:  # NPR
                artifact = self.model.img2npr(x_preprocessed)

            logits = self.model.classify(artifact)

        return logits

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

def create_lgrad_sasv6(
    stylegan_weights: str,
    classifier_weights: str,
    gaussian_sigma: float = 0.8,
    dct_threshold: float = 2.5,
    dwt_threshold: float = 0.05,
    wavelet_type: str = 'haar',
    input_mode: Literal["I_d_wave", "concat", "residual_only"] = "concat",
    residual_type: Literal["R_g", "R_d", "both"] = "R_d",
    device: str = "cuda",
) -> UnifiedSASv6:
    """
    Convenience function to create LGrad model with SASv6 preprocessing

    Args:
        stylegan_weights: Path to StyleGAN discriminator weights
        classifier_weights: Path to ResNet50 classifier weights
        gaussian_sigma: Gaussian smoothing sigma
        dct_threshold: DCT soft-threshold value
        dwt_threshold: DWT soft-threshold value (λ)
        wavelet_type: Wavelet type ('haar', 'db2', etc.)
        input_mode: How to construct input ("I_d_wave", "concat", "residual_only")
        residual_type: Which residual to use ("R_g", "R_d", "both")
        device: Device to use

    Returns:
        UnifiedSASv6 model ready for inference

    Example:
        >>> model = create_lgrad_sasv6(
        ...     stylegan_weights="weights/stylegan.pth",
        ...     classifier_weights="weights/classifier.pth",
        ...     gaussian_sigma=0.8,
        ...     dwt_threshold=0.05
        ... )
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

    # Apply SASv6
    config = SASv6Config(
        model="LGrad",
        gaussian_sigma=gaussian_sigma,
        dct_threshold=dct_threshold,
        dwt_threshold=dwt_threshold,
        wavelet_type=wavelet_type,
        input_mode=input_mode,
        residual_type=residual_type,
        device=device,
    )

    model = UnifiedSASv6(lgrad, config)

    return model


def create_npr_sasv6(
    weights: str,
    gaussian_sigma: float = 0.8,
    dct_threshold: float = 2.5,
    dwt_threshold: float = 0.05,
    wavelet_type: str = 'haar',
    input_mode: Literal["I_d_wave", "concat", "residual_only"] = "concat",
    residual_type: Literal["R_g", "R_d", "both"] = "R_d",
    device: str = "cuda",
) -> UnifiedSASv6:
    """
    Convenience function to create NPR model with SASv6 preprocessing

    Args:
        weights: Path to NPR model weights
        gaussian_sigma: Gaussian smoothing sigma
        dct_threshold: DCT soft-threshold value
        dwt_threshold: DWT soft-threshold value (λ)
        wavelet_type: Wavelet type ('haar', 'db2', etc.)
        input_mode: How to construct input ("I_d_wave", "concat", "residual_only")
        residual_type: Which residual to use ("R_g", "R_d", "both")
        device: Device to use

    Returns:
        UnifiedSASv6 model ready for inference

    Example:
        >>> model = create_npr_sasv6(
        ...     weights="weights/npr.pth",
        ...     gaussian_sigma=0.8,
        ...     dwt_threshold=0.05
        ... )
        >>> probs = model.predict_proba(images)
    """
    from model.NPR.npr_model import NPR

    # Create base NPR model
    npr = NPR(
        weights=weights,
        device=device,
    )

    # Apply SASv6
    config = SASv6Config(
        model="NPR",
        gaussian_sigma=gaussian_sigma,
        dct_threshold=dct_threshold,
        dwt_threshold=dwt_threshold,
        wavelet_type=wavelet_type,
        input_mode=input_mode,
        residual_type=residual_type,
        device=device,
    )

    model = UnifiedSASv6(npr, config)

    return model
