"""Training and evaluation utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any, Callable
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
import json
from pathlib import Path


class MetricsLogger:
    """Logger for tracking metrics across training steps."""

    def __init__(self):
        """Initialize metrics logger."""
        self.metrics = {}
        self.step_metrics = {}

    def log_step(self, split: str, **kwargs):
        """Log metrics for a step.

        Args:
            split: Dataset split (e.g., 'train', 'valid')
            **kwargs: Metric name -> value pairs
        """
        if split not in self.metrics:
            self.metrics[split] = {k: [] for k in kwargs.keys()}

        for key, value in kwargs.items():
            if key not in self.metrics[split]:
                self.metrics[split][key] = []
            self.metrics[split][key].append(value)

        self.step_metrics = {f"{split}_{k}": v for k, v in kwargs.items()}

    def latest(self, metric_names: list) -> str:
        """Get latest values for specified metrics.

        Args:
            metric_names: List of metric names to retrieve

        Returns:
            Formatted string of latest values
        """
        parts = []
        for name in metric_names:
            if name in self.step_metrics:
                value = self.step_metrics[name]
                if isinstance(value, float):
                    parts.append(f"{name}={value:.4f}")
                else:
                    parts.append(f"{name}={value}")
        return " | ".join(parts)

    def flush(self, step: int = None):
        """Flush step metrics.

        Args:
            step: Optional step number to include in logging
        """
        self.step_metrics = {}

    def export(self) -> Dict[str, Any]:
        """Export all metrics.

        Returns:
            Dictionary of metrics for all splits
        """
        return self.metrics

    def to_df(self):
        """Convert metrics to pandas DataFrame.

        Returns:
            DataFrame with metrics
        """
        try:
            import pandas as pd

            data = []
            for split, metrics in self.metrics.items():
                for step, values in enumerate(zip(*metrics.values())):
                    row = {"split": split, "step": step}
                    for metric_name, value in zip(metrics.keys(), values):
                        row[metric_name] = value
                    data.append(row)
            return pd.DataFrame(data)
        except ImportError:
            raise ImportError("pandas is required for to_df()")


def calculate_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate loss and logits for a batch.

    Args:
        model: PyTorch model
        batch: Dictionary with 'sequences' and 'labels'
        criterion: Loss function
        device: Device to run on

    Returns:
        Tuple of (loss, logits)
    """
    sequences = batch["sequences"].to(device)
    labels = batch["labels"].to(device)

    logits = model(sequences)
    loss = criterion(logits, labels)

    return loss, logits


@torch.no_grad()
def compute_metrics(
    y_true: np.ndarray, logits: np.ndarray
) -> Dict[str, float]:
    """Compute accuracy and auROC for model predictions.

    Args:
        y_true: True labels (binary)
        logits: Model logits

    Returns:
        Dictionary with 'accuracy' and 'auc' metrics
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= 0.5).astype(int).flatten()
    y_true_flat = y_true.flatten()

    accuracy = accuracy_score(y_true_flat, preds)
    auc = roc_auc_score(y_true_flat, logits.flatten())

    return {
        "accuracy": accuracy,
        "auc": auc,
    }


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, Dict[str, float]]:
    """Run a single training step.

    Args:
        model: PyTorch model in training mode
        batch: Batch of data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
        scaler: Optional gradient scaler for mixed precision

    Returns:
        Tuple of (loss_value, metrics)
    """
    optimizer.zero_grad()

    if scaler is not None:
        with torch.cuda.amp.autocast():
            loss, logits = calculate_loss(model, batch, criterion, device)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss, logits = calculate_loss(model, batch, criterion, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return loss.item(), {}


@torch.no_grad()
def eval_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a single batch.

    Args:
        model: PyTorch model in eval mode
        batch: Batch of data
        criterion: Loss function
        device: Device to run on

    Returns:
        Dictionary with evaluation metrics
    """
    loss, logits = calculate_loss(model, batch, criterion, device)

    labels = batch["labels"].to(device)
    metrics = compute_metrics(labels.cpu().numpy(), logits.cpu().numpy())
    metrics["loss"] = loss.item()

    return metrics


def train(
    model: nn.Module,
    dataset_splits: Dict[str, DataLoader],
    num_epochs: int,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
    class_weights: Optional[torch.Tensor] = None,
    eval_every: int = 100,
    save_best: bool = False,
    checkpoint_dir: Optional[str] = None,
    use_mixed_precision: bool = False,
) -> Tuple[nn.Module, MetricsLogger]:
    """Train a model.

    Args:
        model: PyTorch model
        dataset_splits: Dictionary with 'train' and 'valid' DataLoaders
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on (default: cuda if available, else cpu)
        class_weights: Optional class weights for loss function
        eval_every: Evaluate every N batches
        save_best: Whether to save best model
        checkpoint_dir: Directory to save checkpoints
        use_mixed_precision: Whether to use mixed precision training

    Returns:
        Tuple of (trained_model, metrics_logger)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()

    # Loss function
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(dataset_splits["train"])
    )

    # Gradient scaler for mixed precision
    scaler = (
        torch.cuda.amp.GradScaler() if use_mixed_precision else None
    )

    metrics_logger = MetricsLogger()
    best_auc = 0.0
    global_step = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        pbar = tqdm(
            dataset_splits["train"], desc=f"Epoch {epoch + 1}/{num_epochs}"
        )

        for batch_idx, batch in enumerate(pbar):
            loss, _ = train_step(
                model, batch, optimizer, criterion, device, scaler
            )
            scheduler.step()
            metrics_logger.log_step(split="train", loss=loss)

            pbar.set_postfix_str(metrics_logger.latest(["loss"]))
            global_step += 1

            # Validation
            if global_step % eval_every == 0:
                model.eval()
                all_labels = []
                all_logits = []

                with torch.no_grad():
                    for val_batch in dataset_splits["valid"]:
                        val_loss, val_logits = calculate_loss(
                            model, val_batch, criterion, device
                        )
                        val_labels = val_batch["labels"].to(device)

                        all_labels.append(val_labels.cpu().numpy())
                        all_logits.append(val_logits.cpu().numpy())

                        metrics_logger.log_step(split="valid", loss=val_loss.item())

                # Compute overall metrics
                all_labels = np.concatenate(all_labels)
                all_logits = np.concatenate(all_logits)
                overall_metrics = compute_metrics(all_labels, all_logits)

                metrics_logger.log_step(split="valid", **overall_metrics)
                metrics_logger.flush(step=global_step)

                print(
                    f"\nStep {global_step} - "
                    f"Train loss: {metrics_logger.metrics['train']['loss'][-1]:.4f} | "
                    f"Val loss: {metrics_logger.metrics['valid']['loss'][-1]:.4f} | "
                    f"Val AUC: {overall_metrics['auc']:.4f}"
                )

                # Save best model
                if save_best and overall_metrics["auc"] > best_auc:
                    best_auc = overall_metrics["auc"]
                    if checkpoint_dir:
                        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                        checkpoint_path = Path(checkpoint_dir) / "best_model.pt"
                        torch.save(model.state_dict(), checkpoint_path)
                        print(f"Saved best model (AUC: {best_auc:.4f})")

                model.train()

    return model, metrics_logger


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluate model on test set.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Dictionary with test metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            sequences = batch["sequences"].to(device)
            labels = batch["labels"].to(device)

            logits = model(sequences)

            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_logits = np.concatenate(all_logits)

    # Compute metrics
    metrics = compute_metrics(all_labels, all_logits)

    # Additional metrics
    probs = 1 / (1 + np.exp(-all_logits))
    fpr, tpr, _ = roc_curve(all_labels, all_logits)
    precision, recall, _ = precision_recall_curve(all_labels, all_logits)

    metrics["fpr"] = fpr.tolist()
    metrics["tpr"] = tpr.tolist()
    metrics["precision"] = precision.tolist()
    metrics["recall"] = recall.tolist()

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    path: str,
):
    """Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Metrics dictionary
        path: Path to save checkpoint
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, path: str
) -> Tuple[int, Dict[str, Any]]:
    """Load model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        path: Path to checkpoint

    Returns:
        Tuple of (epoch, metrics)
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    metrics = checkpoint.get("metrics", {})
    return epoch, metrics
