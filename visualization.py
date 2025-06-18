"""Visualization utilities for model analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path


# Default colors for plotting
DEFAULT_SPLIT_COLORS = {
    "train": "#1f77b4",      # blue
    "valid": "#ff7f0e",      # orange
    "test": "#2ca02c",       # green
}


def plot_binding_site(
    sequence: np.ndarray,
    importance_matrix: np.ndarray,
    sequence_str: Optional[str] = None,
    title: str = "Binding Site Importance",
    figsize: Tuple[int, int] = (12, 4),
    cmap: str = "RdBu_r",
    save_path: Optional[str] = None,
):
    """Plot sequence with importance heatmap (ISM or gradients).

    Args:
        sequence: One-hot encoded sequence of shape (seq_len, 4)
        importance_matrix: Importance matrix of shape (seq_len, 4)
        sequence_str: Optional DNA sequence string
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        importance_matrix.T,
        cmap=cmap,
        center=0,
        ax=ax,
        cbar_kws={"label": "Importance"},
        xticklabels=10,  # Show every 10th position
    )

    # Set y-axis labels to DNA bases
    ax.set_yticklabels(["A", "C", "G", "T"], rotation=0)
    ax.set_xlabel("Sequence Position")
    ax.set_ylabel("DNA Base")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_10_gradients(
    sequences: np.ndarray,
    gradient_matrices: List[np.ndarray],
    sequence_strs: Optional[List[str]] = None,
    title: str = "Input Gradients",
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
):
    """Plot multiple gradient matrices in a grid.

    Args:
        sequences: List or array of sequences
        gradient_matrices: List of gradient matrices
        sequence_strs: Optional list of sequence strings
        title: Plot title
        figsize: Figure size (default: auto-computed)
        save_path: Optional path to save figure
    """
    num_seqs = len(gradient_matrices)
    num_cols = min(5, num_seqs)
    num_rows = (num_seqs + num_cols - 1) // num_cols

    if figsize is None:
        figsize = (num_cols * 3, num_rows * 2.5)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_seqs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, grad_matrix in enumerate(gradient_matrices):
        ax = axes[i]

        sns.heatmap(
            grad_matrix.T,
            cmap="RdBu_r",
            center=0,
            ax=ax,
            cbar=False,
        )

        ax.set_yticklabels(["A", "C", "G", "T"], rotation=0)
        ax.set_xticklabels([])
        if sequence_strs and i < len(sequence_strs):
            ax.set_title(f"Seq {i}: {sequence_strs[i][:20]}...")
        else:
            ax.set_title(f"Sequence {i}")

    # Hide extra subplots
    for i in range(num_seqs, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.00)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axes


def plot_learning(
    metrics: Dict[str, Any],
    metric_name: str = "loss",
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
):
    """Plot learning curves.

    Args:
        metrics: Dictionary with metrics from training
        metric_name: Name of metric to plot ('loss', 'accuracy', 'auc')
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for split, split_metrics in metrics.items():
        if metric_name in split_metrics:
            values = split_metrics[metric_name]
            color = DEFAULT_SPLIT_COLORS.get(split, "black")
            ax.plot(values, label=split, color=color, linewidth=2)

    ax.set_xlabel("Step")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f"{metric_name.capitalize()} During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_roc_curve(
    y_true: np.ndarray,
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
):
    """Plot ROC curve.

    Args:
        y_true: True labels
        fpr: False positive rates
        tpr: True positive rates
        auc: AUC score
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_precision_recall_curve(
    y_true: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
):
    """Plot precision-recall curve.

    Args:
        y_true: True labels
        precision: Precision values
        recall: Recall values
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(recall, precision, color="darkorange", lw=2, label="Precision-Recall")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_position_importance(
    position_importance: np.ndarray,
    sequence_str: Optional[str] = None,
    title: str = "Position Importance",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
):
    """Plot position-level importance scores.

    Args:
        position_importance: Importance scores of shape (seq_len,)
        sequence_str: Optional DNA sequence string to display
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    positions = np.arange(len(position_importance))
    ax.bar(positions, position_importance, color="steelblue", alpha=0.7)

    ax.set_xlabel("Sequence Position")
    ax.set_ylabel("Importance")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    if sequence_str and len(sequence_str) <= 200:
        # Add sequence below
        ax.set_xticks(positions[::20])
        bases = list(sequence_str)
        ax.set_xticklabels([bases[i] if i < len(bases) else "" for i in positions[::20]])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_mutations_effect(
    ism_matrix: np.ndarray,
    baseline_prob: float,
    title: str = "Mutation Effects",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
):
    """Plot in silico mutagenesis effects.

    Args:
        ism_matrix: ISM matrix of shape (seq_len, 4)
        baseline_prob: Baseline binding probability
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 3]})

    # Position-level effects
    position_effects = ism_matrix.max(axis=1)
    positions = np.arange(len(position_effects))
    ax1.bar(positions, position_effects, color="steelblue", alpha=0.7)
    ax1.set_ylabel("Max Effect")
    ax1.set_title(f"{title} (Baseline P={baseline_prob:.3f})")
    ax1.grid(True, alpha=0.3, axis="y")

    # Heatmap
    sns.heatmap(
        ism_matrix.T,
        cmap="RdBu_r",
        center=0,
        ax=ax2,
        cbar_kws={"label": "Δ Probability"},
    )
    ax2.set_yticklabels(["A", "C", "G", "T"], rotation=0)
    ax2.set_xlabel("Sequence Position")
    ax2.set_ylabel("DNA Base")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, (ax1, ax2)


def plot_model_comparison(
    models_metrics: Dict[str, Dict[str, float]],
    metric_names: List[str] = ["accuracy", "auc"],
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
):
    """Plot comparison of multiple models.

    Args:
        models_metrics: Dictionary mapping model names to metric dicts
        metric_names: List of metrics to compare
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, len(metric_names), figsize=figsize)
    if len(metric_names) == 1:
        axes = [axes]

    model_names = list(models_metrics.keys())

    for ax, metric in zip(axes, metric_names):
        values = [models_metrics[m].get(metric, 0) for m in model_names]
        ax.bar(model_names, values, color="steelblue", alpha=0.7)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} Comparison")
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axes


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (6, 5),
    save_path: Optional[str] = None,
):
    """Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax
