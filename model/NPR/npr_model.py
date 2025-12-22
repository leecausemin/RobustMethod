import sys
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class NPR(nn.Module):
    """
    NPR: Neighboring Pixel Relationships for Generalizable Deepfake Detection
    """
    def __init__(
            self,
            weights: Optional[str] = None,
            device: str = "cuda"
    ):
        super().__init__()
        self.device = device

        # Add NPR original repo path for imports and import resnet50
        REPO_PATH = Path(__file__).parent / "npr"
        if str(REPO_PATH) not in sys.path:
            sys.path.insert(0, str(REPO_PATH))

        # Force reimport to avoid cached standard resnet
        if 'networks' in sys.modules:
            del sys.modules['networks']
        if 'networks.resnet' in sys.modules:
            del sys.modules['networks.resnet']

        # Import here to ensure correct path is used
        from networks.resnet import resnet50

        # ResNet50 with NPR
        self.model = resnet50(num_classes=1)

        if weights:
            checkpoint = torch.load(weights, map_location="cpu")
            # Handle both direct state_dict and checkpoint dict formats
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # Remove "module." prefix from DataParallel checkpoints
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict, strict=True)

        # Move model to device
        self.model = self.model.to(device)
        self.to(device)

    def interpolate(self, img, factor):
        """Interpolate helper function (same as NPR ResNet)"""
        return F.interpolate(
            F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True),
            scale_factor=1/factor, mode='nearest', recompute_scale_factor=True
        )

    def img2npr(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract NPR (Neighboring Pixel Relationships) artifact.

        Args:
            x: Input images [B, 3, H, W], range [0, 1]
        Returns:
            NPR artifact [B, 3, H, W]
        """
        # Ensure x is on the correct device
        if x.device != self.device:
            x = x.to(self.device)

        # NPR = x - interpolate(x, 0.5)
        # This captures artifacts from downsampling-upsampling
        npr = x - self.interpolate(x, 0.5)
        return npr

    def classify(self, npr: torch.Tensor) -> torch.Tensor:
        """
        Classify NPR artifact.

        Args:
            npr: NPR artifact [B, 3, H, W]
        Returns:
            logits: [B, 1] (positive = fake)
        """
        # Ensure npr is on the correct device
        if npr.device != self.device:
            npr = npr.to(self.device)

        # The ResNet forward already extracts NPR internally,
        # but since we're passing pre-computed NPR, we need to skip that step.
        # For now, we'll use the full model which will re-extract NPR.
        # TODO: Refactor to use pre-computed NPR directly
        # For compatibility, we pass through the full model
        # which will extract NPR again (inefficient but works)

        # Actually, let's compute from NPR directly
        # We need to call the ResNet layers after NPR extraction
        x = self.model.conv1(npr * 2.0 / 3.0)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.model.fc1(x)

        return logits

    def forward(
        self,
        x: torch.Tensor,
        return_npr: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input images [B, 3, H, W], normalized to [-1, 1] or [0, 1]
            return_npr: If True, also return NPR artifact
        Returns:
            logits: [B, 1] (positive = fake)
            npr (optional): [B, 3, H, W] NPR artifact
        """
        # Ensure x is on the correct device
        if x.device != self.device:
            x = x.to(self.device)

        npr = self.img2npr(x)
        logits = self.classify(npr)

        if return_npr:
            return logits, npr

        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict real/fake labels
        Args:
            x: Input images [B, 3, H, W]
        Returns:
            predictions: [B] (0=real, 1=fake)
        """
        logits = self.forward(x)
        return (torch.sigmoid(logits) > 0.5).long().squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probability of being fake
        Args:
            x: Input image [B, 3, H, W]
        Returns:
            probabilities: [B] (probability of fake)
        """
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(1)