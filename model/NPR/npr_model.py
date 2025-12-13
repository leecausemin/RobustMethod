import sys
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Add NPR original repo path for imports
REPO_PATH = Path(__file__).parent / "npr"
sys.path.insert(0, str(REPO_PATH))

from networks.resnet import resnet50

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

        # ResNet50 with NPR
        self.model = resnet50(num_classes=1)

        if weights:
            self.model.load_state_dict(
                torch.load(weights, map_location="cpu"),
                strict=True
            )
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, 3, H, W], normalized to [-1, 1] or [0, 1]
        Returns:
            logits: [B, 1] (positive = fake)
        """
        logits = self.model(x)
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