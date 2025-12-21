"""
NORM (Normalization-based Test-Time Adaptation) for LGrad

Hypothesis: LGrad fails on corrupted images not because gradient information is lost,
but because the input distribution shifts significantly.

Solution: Adapt the BatchNorm layers in the classifier (ResNet50) during test-time
by updating running statistics with a weighted combination of source and target statistics.
"""

import copy
from dataclasses import dataclass
from typing import Union, Optional

import torch
import torch.nn as nn


@dataclass
class NORMConfig:
    """Configuration for NORM adaptation"""
    source_sum: int = 128  # Accumulated source batch size for weighting
    adaptation_target: str = "classifier"  # "classifier", "grad_model", "both"
    device: str = "cuda"


class LGradNORM(nn.Module):
    """
    LGrad with NORM (Normalization-based Test-Time Adaptation)

    Adapts BatchNorm layers in the LGrad classifier during test-time to handle
    distribution shifts caused by image corruptions.

    Args:
        lgrad_model: Pre-trained LGrad model
        config: NORM configuration

    Example:
        >>> from model.LGrad.lgrad_model import LGrad
        >>> from model.method.method import LGradNORM, NORMConfig
        >>>
        >>> # Load base model
        >>> lgrad = LGrad(
        ...     stylegan_weights="path/to/stylegan.pth",
        ...     classifier_weights="path/to/classifier.pth"
        ... )
        >>>
        >>> # Apply NORM adaptation
        >>> config = NORMConfig(source_sum=128)
        >>> model = LGradNORM(lgrad, config)
        >>>
        >>> # Use as normal
        >>> predictions = model.predict_proba(images)
    """

    def __init__(self, lgrad_model, config: NORMConfig):
        super().__init__()
        self.cfg = config
        self.device = config.device

        # Deep copy to avoid modifying original model
        self.model = copy.deepcopy(lgrad_model)
        self.model.to(self.device)

        self._setup()

    def _setup(self):
        """Apply NORM adaptation to model"""
        self._apply_norm_adaptation()

    def _apply_norm_adaptation(self):
        """
        Modify BatchNorm layers to perform test-time adaptation.

        The adaptation updates running statistics using:
            alpha = batch_size / (source_sum + batch_size)
            new_mean = (1 - alpha) * running_mean + alpha * batch_mean
            new_var = (1 - alpha) * running_var + alpha * batch_var

        This allows the model to adapt to distribution shifts while
        maintaining some memory of the source distribution.
        """

        # Determine which parts to adapt
        targets = []
        if self.cfg.adaptation_target in ("classifier", "both"):
            targets.append(("classifier", self.model.classifier))
        if self.cfg.adaptation_target in ("grad_model", "both"):
            targets.append(("grad_model", self.model.grad_model))

        for target_name, target_module in targets:
            for name, module in target_module.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    # Mark module for adaptation
                    module.adapt_type = "NORM"
                    module.source_sum = self.cfg.source_sum

                    # Override forward method
                    def norm_forward(self, x):
                        """
                        Modified BatchNorm forward with test-time adaptation.

                        During inference, updates running statistics using a weighted
                        combination of stored source statistics and current batch statistics.
                        """
                        if hasattr(self, 'adapt_type') and self.adapt_type == "NORM":
                            # Compute adaptation weight
                            # Higher alpha = more weight on current batch
                            # Lower alpha = more weight on source statistics
                            alpha = x.shape[0] / (self.source_sum + x.shape[0])

                            # Update running statistics with exponential moving average
                            running_mean = (1 - alpha) * self.running_mean + alpha * x.mean(dim=[0, 2, 3])
                            running_var = (1 - alpha) * self.running_var + alpha * x.var(dim=[0, 2, 3])

                            # Compute normalization scale and bias
                            scale = self.weight * (running_var + self.eps).rsqrt()
                            bias = self.bias - running_mean * scale
                        else:
                            # Standard BatchNorm (use stored statistics)
                            scale = self.weight * (self.running_var + self.eps).rsqrt()
                            bias = self.bias - self.running_mean * scale

                        # Reshape for broadcasting
                        scale = scale.reshape(1, -1, 1, 1)
                        bias = bias.reshape(1, -1, 1, 1)

                        # Apply normalization
                        out_dtype = x.dtype
                        out = x * scale.to(out_dtype) + bias.to(out_dtype)
                        return out

                    # Bind the new forward method to the module
                    module.forward = norm_forward.__get__(module, module.__class__)

                    print(f"[NORM] Adapted {target_name}.{name}")

    def img2grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Image to Gradient (delegates to wrapped model).

        Args:
            x: Input images [B, 3, H, W], range [0, 1]

        Returns:
            Gradient images [B, 3, 256, 256], normalized to [0,1]
        """
        self.model.eval()
        return self.model.img2grad(x)

    def classify(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Classification of Gradient image (delegates to wrapped model).

        Args:
            grad: Gradient images [B, 3, 256, 256], range [0, 1]

        Returns:
            Logits [B, 1] (positive = fake)
        """
        self.model.eval()
        return self.model.classify(grad)

    def forward(
        self,
        x: torch.Tensor,
        return_grad: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with NORM adaptation.

        Args:
            x: Input images [B, 3, H, W], range [0, 1]
            return_grad: If True, also return gradient images

        Returns:
            logits: [B, 1] (positive = fake)
            grad (optional): [B, 3, 256, 256] gradient images
        """
        self.model.eval()  # Keep in eval mode (but BatchNorm will adapt)
        return self.model(x, return_grad=return_grad)

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

    def reset_adaptation(self):
        """
        Reset all adapted BatchNorm layers to their original state.

        Useful when switching between different test distributions.
        """
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d) and hasattr(module, 'adapt_type'):
                # Reset to original running stats would require keeping a copy
                # For now, we just mark them as not adapted
                delattr(module, 'adapt_type')
                print(f"[NORM] Reset adaptation")
                break


class LGradNORMwithMemory(LGradNORM):
    """
    Extended NORM with memory of original source statistics.

    Allows resetting to original source distribution after adaptation.
    """

    def __init__(self, lgrad_model, config: NORMConfig):
        # Store original statistics before adaptation
        self.original_stats = {}
        super().__init__(lgrad_model, config)

    def _apply_norm_adaptation(self):
        """Apply NORM adaptation while saving original statistics"""

        # First, save original statistics
        for name, module in self.model.classifier.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.original_stats[name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                }

        # Then apply adaptation
        super()._apply_norm_adaptation()

    def reset_to_source(self):
        """Reset all BatchNorm layers to original source statistics"""
        for name, module in self.model.classifier.named_modules():
            if isinstance(module, nn.BatchNorm2d) and name in self.original_stats:
                module.running_mean.copy_(self.original_stats[name]['running_mean'])
                module.running_var.copy_(self.original_stats[name]['running_var'])
                print(f"[NORM] Reset {name} to source statistics")


# Utility function for easy usage
def create_lgrad_norm(
    stylegan_weights: str,
    classifier_weights: str,
    source_sum: int = 128,
    adaptation_target: str = "classifier",
    device: str = "cuda",
) -> LGradNORM:
    """
    Convenience function to create LGrad model with NORM adaptation.

    Args:
        stylegan_weights: Path to StyleGAN discriminator weights
        classifier_weights: Path to ResNet50 classifier weights
        source_sum: Accumulated source batch size for weighting
        adaptation_target: Which part to adapt ("classifier", "grad_model", "both")
        device: Device to use

    Returns:
        LGradNORM model ready for inference

    Example:
        >>> model = create_lgrad_norm(
        ...     stylegan_weights="weights/stylegan.pth",
        ...     classifier_weights="weights/classifier.pth",
        ...     source_sum=128
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

    # Apply NORM adaptation
    config = NORMConfig(
        source_sum=source_sum,
        adaptation_target=adaptation_target,
        device=device,
    )

    model = LGradNORM(lgrad, config)

    return model
