import sys
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torchvision import transforms

# Add LGrad original repo path for imports
REPO_PATH = Path(__file__).parent / "lgrad"
sys.path.insert(0, str(REPO_PATH / "img2gad_pytorch"))
sys.path.insert(0, str(REPO_PATH / "CNNDetection"))

from models import build_model
from networks.resnet import resnet50

class LGrad(nn.Module):
    """
    LGrad: Learning on Gradients for GAN-Generated Image Detection
    """
    def __init__(
            self,
            stylegan_weights: Optional[str] = None,
            classifier_weights: Optional[str] = None,
            device: str = "cuda",
            resize: int = 256,
    ):
        super().__init__()
        self.device = device
        self.resize = resize

        # StyleGAN Discriminator (for img2grad)
        self.grad_model = build_model(
            gan_type="stylegan",
            module="discriminator",
            resolution=256,
            label_size=0,
            image_channels=3,
        )

        if stylegan_weights:
            self.grad_model.load_state_dict(
                torch.load(stylegan_weights, map_location="cpu"),
                strict=True
            )

        # ResNet50 Classifier
        self.classifier = resnet50(num_classes=1)

        if classifier_weights:
            self.classifier.load_state_dict(
                torch.load(classifier_weights, map_location="cpu")
            )
        
        # Transforms
        self.img2grad_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.grad2clf_transform = transforms.Compose([
            # No resize! Keep 256x256 (same as original LGrad)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.to(device)

    def img2grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Image to Gradient.
        Args:
            x: Input images [B, 3, H, W], range [0, 1]
        Returns:
            Gradient images [B, 3, 256, 256], normalized to [0,1]
        """
        # Enable gradient computation even in eval mode
        with torch.enable_grad():
            x = self.img2grad_transform(x)
            x = x.to(self.device).requires_grad_(True)

            # Forward through discriminator
            out = self.grad_model(x)

            # Compute gradient w.r.t. input
            self.grad_model.zero_grad()
            grad = torch.autograd.grad(
                outputs=out.sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True,
            )[0]

            # Detach gradient to prevent graph retention
            grad = grad.detach()

        # Normalize gradient to [0, 1]
        grad = self._normalize_grad(grad)

        return grad
    
    def _normalize_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Normalize gradient to [0, 1] with uint8 quantization simulation
        (same as original LGrad: save as PNG then reload)
        """
        B = grad.shape[0]

        grad = grad.view(B, -1)

        # Normalize to [0, 255]
        grad_min = grad.min(dim=1, keepdim=True)[0]
        grad_max = grad.max(dim=1, keepdim=True)[0]

        grad = (grad - grad_min) / (grad_max - grad_min + 1e-8)
        grad = grad * 255.0

        # Simulate uint8 quantization (as if saving and reloading PNG)
        grad = torch.round(grad).clamp(0, 255)

        # Convert back to [0, 1]
        grad = grad / 255.0

        grad = grad.view(B, 3, self.resize, self.resize)

        return grad
    
    def classify(self, grad: torch.Tensor) -> torch.Tensor:
        """
        classification of Gradient image
        Args:
            grad: Gradient images [B, 3, 256, 256], range [0, 1]
        Returns:
            Logits [B, 1] (positive = fake)
        """
        x = self.grad2clf_transform(grad)
        logits = self.classifier(x)
        return logits
    
    def forward(
            self,
            x: torch.Tensor,
            return_grad: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input images [B, 3, H, W], range [0, 1]
            return_grad: If True, also return gradient images
        Returns:
            logits: [B, 1] (positive = fake)
            grad (optional): [B, 3, 256, 256] gradient images
        """
        grad = self.img2grad(x)
        logits = self.classify(grad)

        if return_grad:
            return logits, grad
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict real/fake labels
        Args:
            x: Input images [B, 3, H, W], range [0, 1]
        Returns:
            predictions: [B] (0=real, 1=fake)
        """
        logits =self.forward(x)
        return (torch.sigmoid(logits) > 0.5).long().squeeze(1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probability of being fake
        Args:
            x: Input images [B, 3, H, W], range [0, 1]
        Returns:
            probabilities: [B] (probability of fake)
        """
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(1)



