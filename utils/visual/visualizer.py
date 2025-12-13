import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Literal
import random

from utils.data.dataset import CorruptedDataset

class DatasetVisualizer:
    """
    Visualizer for CorruptedDataset
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
    
    def _build_filename_index(self, dataset: CorruptedDataset) -> dict:
        """
        Build index: (dataset, label, filename) -> {corruption: sample_idx}
        """
        index = {}
        for idx, sample in enumerate(dataset.samples):
            key = (sample["dataset"], sample["label"], sample["filename"])

            if key not in index:
                index[key] = {}
            index[key][sample["corruption"]] = idx
        
        return index
    
    def __call__(
            self,
            dataset: CorruptedDataset,
            corruption: str = "all",
            label: Optional[Literal["real", "fake"]] = None,
            dataset_name: Optional[str] = None,
            n_samples: int = 1,
            figsize: Optional[tuple] = None,
    ):
        """
        Visualize images from dataset
        
        Args:
            dataset: CorruptedDataset instance
            corruption: "all" for all corruptions grid, or specific corruption name
            label: filter by "real" or "fake"
            dataset_name: filter by dataset name
            n_samples: number of samples to show(row when corruption="all")
            figsize: figure size (auto if None)
        """
        if corruption == "all":
            self._show_all_corruptions(dataset, label, dataset_name, n_samples, figsize)
        else:
            self._show_single_corruption(dataset, corruption, label, dataset_name, n_samples, figsize)

    def _show_all_corruptions(
            self,
            dataset: CorruptedDataset,
            label: Optional[str],
            dataset_name: Optional[str],
            n_samples: int,
            figsize: Optional[tuple],
    ):
        """
        show same image across all corruptions
        """
        filename_index = self._build_filename_index(dataset)
        corruptions = dataset.corruptions
        label_map = {"real": 0, "fake": 1}

        # Find complete samples (have all corruptions)
        complete_samples = []
        for key, corr_dict in filename_index.items():
            ds, lbl, fname = key
            if len(corr_dict) != len(corruptions):
                continue
            if dataset_name and ds != dataset_name:
                continue
            if label and lbl != label_map[label]:
                continue
            complete_samples.append(key)

        if not complete_samples:
            print("No complete samples found with all corruptions.")
            return
        
        # Select samples
        n_samples = min(n_samples, len(complete_samples))
        selected_keys = random.sample(complete_samples, n_samples)

        # Plot
        n_cols = len(corruptions)
        n_rows = n_samples
        if figsize is None:
            figsize = (3 * n_cols, 3 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes]
        
        for row, key in enumerate(selected_keys):
            ds, lbl, fname = key
            corr_indices = filename_index[key]
            label_str = "FAKE" if lbl == 1 else "REAL"

            for col, corr in enumerate(corruptions):
                idx = corr_indices[corr]
                image, _, metadata = dataset[idx]

                img_np = image.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)

                axes[row][col].imshow(img_np)
                axes[row][col].axis("off")

                if row == 0:
                    axes[row][col].set_title(corr, fontsize=10)
                if col == 0:
                    axes[row][col].set_ylabel(f"[{label_str}]\n{ds}", fontsize=9)
            
        plt.tight_layout()
        plt.show()

    def _show_single_corruption(
        self,
        dataset: CorruptedDataset,
        corruption: str,
        label: Optional[str],
        dataset_name: Optional[str],
        n_samples: int,
        figsize: Optional[tuple],
    ):
        """
        Show images for a single corruption type
        """
        label_map = {"real": 0, "fake": 1}

        # Filter samples
        candidates = []
        for idx, sample in enumerate(dataset.samples):
            if sample["corruption"] != corruption:
                continue
            if dataset_name and sample["dataset"] != dataset_name:
                continue
            if label and sample["label"] != label_map[label]:
                continue
            candidates.append(idx)

        if not candidates:
            print(f"No samples found for corruption='{corruption}'")
            return
        
        n_samples = min(n_samples, len(candidates))
        selected = random.sample(candidates, n_samples)

        # Calculate grid
        n_cols = min(4, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols

        if figsize is None:
            figsize = (4 * n_cols, 4 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_samples == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, idx in enumerate(selected):
            row, col = i // n_cols, i % n_cols
            image, lbl, metadata = dataset[idx]

            img_np = image.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)

            axes[row][col].imshow(img_np)
            axes[row][col].axis("off")

            label_str = "FAKE" if lbl == 1 else "REAL"
            axes[row][col].set_title(f"[{label_str}] {metadata['dataset']}", fontsize=9)

        # Hide empty
        for i in range(n_samples, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row][col].axis("off")

        plt.suptitle(f"Corruption: {corruption}", fontsize=14)
        plt.tight_layout()
        plt.show()

    def stats(self, dataset: CorruptedDataset):
        """Print dataset statistics"""
        print("=" * 50)
        print(f"Total samples: {len(dataset):,}")
        print(f"Datasets: {dataset.datasets}")
        print(f"Corruptions: {dataset.corruptions}")
        print("=" * 50)

        real_count = sum(1 for s in dataset.samples if s["label"] == 0)
        fake_count = len(dataset) - real_count
        print(f"Real: {real_count:,} | Fake: {fake_count:,}")