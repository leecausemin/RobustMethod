"""
FreqNorm: Frequency-Domain Gradient Normalization for LGrad

Corruption은 LGrad의 gradient map이 가지는 주파수 에너지 분포를 왜곡한다.
FreqNorm은 학습 없이(test-time 연산만으로) gradient의 주파수 에너지를
'클린 기준 분포'로 되돌려, 분류기가 원래 학습된 artifact 패턴을 다시 보게 만든다.
"""

import torch
import torch.nn as nn
from typing import Union, Optional
from pathlib import Path


# Frequency band definitions (same as analysis scripts)
FREQUENCY_BANDS = [
    (0, 4, "DC+VeryLow"),
    (4, 16, "Low"),
    (16, 64, "Mid"),
    (64, 128, "High"),
    (128, 256, "VeryHigh"),
]


def create_radial_frequency_mask(shape, r_min, r_max, device="cuda"):
    """
    Create radial frequency mask for FFT.

    Args:
        shape: (H, W) of the frequency domain
        r_min: minimum radius
        r_max: maximum radius
        device: computation device

    Returns:
        mask: boolean tensor [H, W]
    """
    H, W = shape
    cy, cx = H // 2, W // 2
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device) - cy,
        torch.arange(W, dtype=torch.float32, device=device) - cx,
        indexing='ij'
    )
    radius = torch.sqrt(x**2 + y**2)
    mask = (radius >= r_min) & (radius < r_max)
    return mask


class FrequencyNormalizer(nn.Module):
    """
    Frequency-domain gradient normalizer.

    Applies band-wise energy normalization to match reference distribution.
    """

    def __init__(
        self,
        reference_stats: dict,
        rho: float = 0.5,
        alpha_min: float = 0.5,
        alpha_max: float = 2.0,
        device: str = "cuda",
    ):
        """
        Args:
            reference_stats: dict with 'mean', 'std', 'median' band energies
            rho: correction strength (0.5~1.0)
            alpha_min: minimum gain (prevent over-darkening)
            alpha_max: maximum gain (prevent over-brightening)
            device: computation device
        """
        super().__init__()

        self.device = device
        self.rho = rho
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Use mean as reference
        self.E_ref = reference_stats['mean'].to(device)

        # Precompute band masks
        H, W = 256, 256
        self.band_masks = []
        for r_min, r_max, band_name in FREQUENCY_BANDS:
            mask = create_radial_frequency_mask((H, W), r_min, r_max, device=device)
            self.band_masks.append(mask)

    def forward(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency normalization to gradient.

        Only adjusts High and VeryHigh frequency bands to avoid
        changing the overall brightness and low-frequency structure.

        Args:
            grad: [B, 3, H, W] gradient tensor

        Returns:
            grad_corrected: [B, 3, H, W] normalized gradient
        """
        B, C, H, W = grad.shape

        # FFT (per-sample processing)
        grad_fft = torch.fft.fft2(grad)  # [B, 3, H, W]
        grad_fft_shifted = torch.fft.fftshift(grad_fft, dim=(-2, -1))

        # Apply band-wise gain (only to High and VeryHigh frequencies)
        # Band indices: 0=DC+VeryLow, 1=Low, 2=Mid, 3=High, 4=VeryHigh
        HIGH_FREQ_BANDS = [3, 4]  # Only adjust High and VeryHigh

        for band_idx, mask in enumerate(self.band_masks):
            # Skip low and mid frequency bands
            if band_idx not in HIGH_FREQ_BANDS:
                continue

            # Compute current energy for this band
            power = torch.abs(grad_fft_shifted) ** 2  # [B, 3, H, W]

            # Average energy per sample
            mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

            if mask.sum() == 0:
                continue

            # Energy per sample: [B]
            energy_per_sample = (power * mask_expanded).sum(dim=(-2, -1, 1)) / mask.sum()

            # Compute gain
            gain = (self.E_ref[band_idx] / (energy_per_sample + 1e-8)) ** self.rho
            gain = torch.clamp(gain, self.alpha_min, self.alpha_max)

            # Apply gain
            gain = gain.view(B, 1, 1, 1)  # [B, 1, 1, 1]
            grad_fft_shifted = torch.where(
                mask_expanded,
                grad_fft_shifted * gain,
                grad_fft_shifted
            )

        # IFFT
        grad_fft_corrected = torch.fft.ifftshift(grad_fft_shifted, dim=(-2, -1))
        grad_corrected = torch.fft.ifft2(grad_fft_corrected).real

        return grad_corrected


class LGradFreqNorm(nn.Module):
    """
    LGrad with Frequency Normalization.

    Applies FreqNorm to gradient before classification.

    Example:
        >>> from model.LGrad.lgrad_model import LGrad
        >>> from model.method.freq_norm import LGradFreqNorm
        >>>
        >>> # Load base model
        >>> lgrad = LGrad(...)
        >>>
        >>> # Load reference stats
        >>> reference_stats = torch.load("freq_reference_progan.pth", weights_only=False)
        >>>
        >>> # Apply FreqNorm
        >>> model = LGradFreqNorm(lgrad, reference_stats, rho=0.5)
        >>>
        >>> # Use as normal
        >>> predictions = model.predict_proba(images)
    """

    def __init__(
        self,
        lgrad_model,
        reference_stats: dict,
        rho: float = 0.5,
        alpha_min: float = 0.5,
        alpha_max: float = 2.0,
        device: str = "cuda",
    ):
        """
        Args:
            lgrad_model: Pre-trained LGrad model
            reference_stats: Reference frequency statistics
            rho: correction strength
            alpha_min: minimum gain
            alpha_max: maximum gain
            device: computation device
        """
        super().__init__()

        self.lgrad = lgrad_model
        self.device = device

        # Frequency normalizer
        self.freq_normalizer = FrequencyNormalizer(
            reference_stats=reference_stats,
            rho=rho,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            device=device,
        )

        self.to(device)

    def img2grad_with_freqnorm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Image to Gradient with FreqNorm.

        Args:
            x: Input images [B, 3, H, W], range [0, 1]

        Returns:
            Gradient images [B, 3, 256, 256], freq-normalized
        """
        # Original gradient
        grad = self.lgrad.img2grad(x)

        # Frequency normalization
        grad_normalized = self.freq_normalizer(grad)

        return grad_normalized

    def forward(
        self,
        x: torch.Tensor,
        return_grad: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with FreqNorm.

        Args:
            x: Input images [B, 3, H, W], range [0, 1]
            return_grad: If True, also return gradient images

        Returns:
            logits: [B, 1] (positive = fake)
            grad (optional): [B, 3, 256, 256] normalized gradient images
        """
        # Get frequency-normalized gradient
        grad = self.img2grad_with_freqnorm(x)

        # Classify
        logits = self.lgrad.classify(grad)

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
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(1)


# Utility function for easy usage
def create_lgrad_freqnorm(
    stylegan_weights: str,
    classifier_weights: str,
    reference_stats_path: str,
    rho: float = 0.3,
    alpha_min: float = 0.85,
    alpha_max: float = 1.15,
    device: str = "cuda",
) -> LGradFreqNorm:
    """
    Convenience function to create LGrad model with FreqNorm.

    Args:
        stylegan_weights: Path to StyleGAN discriminator weights
        classifier_weights: Path to ResNet50 classifier weights
        reference_stats_path: Path to freq_reference_progan.pth
        rho: correction strength
        alpha_min: minimum gain
        alpha_max: maximum gain
        device: Device to use

    Returns:
        LGradFreqNorm model ready for inference

    Example:
        >>> model = create_lgrad_freqnorm(
        ...     stylegan_weights="weights/stylegan.pth",
        ...     classifier_weights="weights/classifier.pth",
        ...     reference_stats_path="freq_reference_progan.pth",
        ...     rho=0.5
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

    # Load reference statistics
    reference_stats = torch.load(reference_stats_path, weights_only=False)

    # Apply FreqNorm
    model = LGradFreqNorm(
        lgrad_model=lgrad,
        reference_stats=reference_stats,
        rho=rho,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        device=device,
    )

    return model
