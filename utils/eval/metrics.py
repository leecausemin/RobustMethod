import numpy as np
from typing import Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class PredictionCollector:
    """
    A class for collecting and storing model outputs.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.labels = []
        self.probs = []
        self.preds = []

    def update(self, labels, probs, threshold=0.5):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()

        self.labels.extend(labels.tolist())
        self.probs.extend(probs.tolist())
        self.preds.extend((probs > threshold).astype(int).tolist())

    def get_results(self):
        """
        Tensor to numpy array
        """
        return {
            "labels": np.array(self.labels),
            "probs": np.array(self.probs),
            "preds": np.array(self.preds)
        }

    def __len__(self):
        return len(self.labels)

class MetricsCalculator:
    """
    A class for computing evaluation metrics.
    """
    def __init__(self):
        from sklearn.metrics import(
            accuracy_score,
            roc_auc_score,
            average_precision_score,
            f1_score,
            recall_score,
        )

        self.results_history = []

        from sklearn.metrics import precision_score

        self.accuracy_score = accuracy_score
        self.roc_auc_score = roc_auc_score
        self.average_precision_score = average_precision_score
        self.f1_score = f1_score
        self.precision_score = precision_score
        self.recall_score = recall_score

    def compute(self, labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
        """
        Args:
            labels: ground truth [N] (0=real, 1=fake)
            probs: predicted probabilities [N] (0~1)
            threshold: classification threshold
        Returns:
            dict with accuracy, auc, ap, f1, precision, recall
        """
        preds = (probs > threshold).astype(int)

        return {
            "accuracy": self.accuracy_score(labels, preds),
            "auc": self.roc_auc_score(labels, probs),
            "ap": self.average_precision_score(labels, probs),
            "f1": self.f1_score(labels, preds),
            "precision": self.precision_score(labels, preds),
            "recall": self.recall_score(labels, preds),
        }
    
    def compute_from_collector(self, collector: PredictionCollector, name: str = "") -> dict:
        """
        Compute from PredictionCollector
        """
        results = collector.get_results()
        metrics = self.compute(results["labels"], results["probs"])
        
        if name:
            self.results_history.append({"name": name, **metrics})
        return metrics

    def print_table(self, metrics: dict, name: str=""):
        """
        Print result
        """
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Evaluation Results")

        table.add_column("Corruption", style="cyan", justify="left")
        table.add_column("Accuracy", justify="right")
        table.add_column("AUC", justify="right")
        table.add_column("AP", justify="right")
        table.add_column("F1", justify="right")

        for r in self.results_history:
            table.add_row(
                r["name"],
                f"{r['accuracy']*100:.2f}%",
                f"{r['auc']*100:.2f}%",
                f"{r['ap']*100:.2f}%",
                f"{r['f1']*100:.2f}%",
            )
        
        console.print(table)

    def reset_history(self):
        """Rest history"""
        self.results_history = []

    def evaluate(
        self,
        model,
        dataloader: DataLoader,
        device: str = "cuda",
        name: str = "",
    ) -> dict:
        """
        Evaluate model on a dataloader

        Args:
            model: PyTorch model
            dataloader: DataLoader instance
            device: device to use
            name: optional name for storing in history

        Returns:
            dict: metrics (accuracy, auc, ap, f1, precision, recall)
        """
        model.eval()
        model.to(device)

        collector = PredictionCollector()

        for batch in tqdm(dataloader, desc=name or "Evaluating", leave=True):
            # Handle different batch formats
            if len(batch) == 3:
                images, labels, metadata = batch
            else:
                images, labels = batch

            images = images.to(device)
            # Enable gradient for input if model requires it (e.g., LGrad)
            images.requires_grad = True

            # Forward pass (model.eval() prevents parameter updates)
            outputs = model(images)

            # Handle different output formats
            if outputs.min() < 0 or outputs.max() > 1:
                probs = torch.sigmoid(outputs).squeeze()
            else:
                probs = outputs.squeeze()

            collector.update(labels, probs.detach())

        # Compute metrics
        metrics = self.compute_from_collector(collector, name=name)
        return metrics

    def print_results_table(self, results: dict = None):
        """
        Print evaluation results in a formatted table

        Args:
            results: dict from evaluate_by_combination() or use self.results_history
        """
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Evaluation Results by Dataset-Corruption")

        table.add_column("Dataset", style="cyan", justify="left")
        table.add_column("Corruption", style="magenta", justify="left")
        table.add_column("Accuracy", justify="right")
        table.add_column("AUC", justify="right")
        table.add_column("AP", justify="right")
        table.add_column("F1", justify="right")

        if results:
            for (dataset_name, corruption), metrics in results.items():
                table.add_row(
                    dataset_name,
                    corruption,
                    f"{metrics['accuracy']*100:.2f}%",
                    f"{metrics['auc']*100:.2f}%",
                    f"{metrics['ap']*100:.2f}%",
                    f"{metrics['f1']*100:.2f}%",
                )
        else:
            # Use results_history
            for r in self.results_history:
                # Parse name as "dataset-corruption"
                parts = r["name"].split("-", 1)
                if len(parts) == 2:
                    dataset_name, corruption = parts
                else:
                    dataset_name = r["name"]
                    corruption = ""

                table.add_row(
                    dataset_name,
                    corruption,
                    f"{r['accuracy']*100:.2f}%",
                    f"{r['auc']*100:.2f}%",
                    f"{r['ap']*100:.2f}%",
                    f"{r['f1']*100:.2f}%",
                )

        console.print(table)

    def summarize_by_corruption(self, results: dict):
        """
        Summarize results by corruption type (average across datasets)

        Args:
            results: dict from evaluate_by_combination()
        """
        from rich.console import Console
        from rich.table import Table

        # Group by corruption
        corruption_metrics = {}
        for (dataset_name, corruption), metrics in results.items():
            if corruption not in corruption_metrics:
                corruption_metrics[corruption] = {
                    "accuracy": [],
                    "auc": [],
                    "ap": [],
                    "f1": [],
                }
            for key in ["accuracy", "auc", "ap", "f1"]:
                corruption_metrics[corruption][key].append(metrics[key])

        # Compute averages
        console = Console()
        table = Table(title="Average Results by Corruption Type")

        table.add_column("Corruption", style="magenta", justify="left")
        table.add_column("Accuracy", justify="right")
        table.add_column("AUC", justify="right")
        table.add_column("AP", justify="right")
        table.add_column("F1", justify="right")

        for corruption, metrics in corruption_metrics.items():
            table.add_row(
                corruption,
                f"{np.mean(metrics['accuracy'])*100:.2f}%",
                f"{np.mean(metrics['auc'])*100:.2f}%",
                f"{np.mean(metrics['ap'])*100:.2f}%",
                f"{np.mean(metrics['f1'])*100:.2f}%",
            )

        console.print(table)

    def summarize_by_dataset(self, results: dict):
        """
        Summarize results by dataset (average across corruptions)

        Args:
            results: dict from evaluate_by_combination()
        """
        from rich.console import Console
        from rich.table import Table

        # Group by dataset
        dataset_metrics = {}
        for (dataset_name, corruption), metrics in results.items():
            if dataset_name not in dataset_metrics:
                dataset_metrics[dataset_name] = {
                    "accuracy": [],
                    "auc": [],
                    "ap": [],
                    "f1": [],
                }
            for key in ["accuracy", "auc", "ap", "f1"]:
                dataset_metrics[dataset_name][key].append(metrics[key])

        # Compute averages
        console = Console()
        table = Table(title="Average Results by Dataset")

        table.add_column("Dataset", style="cyan", justify="left")
        table.add_column("Accuracy", justify="right")
        table.add_column("AUC", justify="right")
        table.add_column("AP", justify="right")
        table.add_column("F1", justify="right")

        for dataset_name, metrics in dataset_metrics.items():
            table.add_row(
                dataset_name,
                f"{np.mean(metrics['accuracy'])*100:.2f}%",
                f"{np.mean(metrics['auc'])*100:.2f}%",
                f"{np.mean(metrics['ap'])*100:.2f}%",
                f"{np.mean(metrics['f1'])*100:.2f}%",
            )

        console.print(table)
