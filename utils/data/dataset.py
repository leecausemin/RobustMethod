import os
from pathlib import Path 
from typing import Optional, Literal

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CorruptedDataset(Dataset):
    DATASETS = ["DiffusionForensics", "ForenSynths", "GANGen", "UniversalFake"]
    CORRUPTIONS = ["original", "contrast", "fog", "gaussian_noise", "jpeg_compression", "motion_blur", "pixelate"]
    LABELS = {"real": 0, "fake": 1}

    def __init__(self,
                 root: str="/workspace/robust_deepfake_ai/dataset",
                 datasets: Optional[list[str]]=None,
                 corruptions: Optional[list[str]]=None,
                 transform: Optional[transforms.Compose]=None,
                 ):
        
        self.root = Path(root)
        self.datasets = datasets or self.DATASETS
        self.corruptions = corruptions or self.CORRUPTIONS
        self.transform = transform or self._default_transform()

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        for dataset_name in self.datasets:
            for corruption in self.corruptions:
                base_path = self.root/dataset_name/corruption

                if not base_path.exists():
                    continue

                # real/fake 폴더 처리
                for label_name, label in self.LABELS.items():
                    label_path = base_path / label_name
                    if label_path.exists():
                        for img_path in label_path.iterdir():
                            if img_path.suffix.lower() in [".png", ".jpg", "jpeg"]:
                                # Skip empty or corrupted files
                                if img_path.stat().st_size == 0:
                                    print(f"Warning: Skipping empty file: {img_path}")
                                    continue

                                self.samples.append({
                                    "path": img_path,
                                    "label": label,
                                    "dataset": dataset_name,
                                    "corruption": corruption,
                                    "filename": img_path.name,
                                })
                
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict]:
        sample = self.samples[idx]

        try:
            image = Image.open(sample["path"]).convert("RGB")
        except Exception as e:
            print(f"Error loading image {sample['path']}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform:
            image = self.transform(image)

        metadata = {
            "path": str(sample["path"]),
            "dataset": sample["dataset"],
            "corruption": sample["corruption"],
        }

        return image, sample["label"], metadata

    def get_combination_counts(self) -> dict:
        """
        Get sample counts for each dataset-corruption combination
        Returns:
            dict: {(dataset, corruption): count}
        """
        counts = {}
        for sample in self.samples:
            key = (sample["dataset"], sample["corruption"])
            counts[key] = counts.get(key, 0) + 1
        return counts

    def get_combinations(self) -> list[tuple[str, str]]:
        """
        Get all dataset-corruption combinations in order
        Returns:
            list of (dataset, corruption) tuples
        """
        combinations = []
        for dataset_name in self.datasets:
            for corruption in self.corruptions:
                key = (dataset_name, corruption)
                if key in self.get_combination_counts():
                    combinations.append(key)
        return combinations
