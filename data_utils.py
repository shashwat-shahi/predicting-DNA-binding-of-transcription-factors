"""Data loading and preprocessing utilities for DNA sequence analysis."""

import numpy as np
import pandas as pd
from typing import Dict
import torch
from torch.utils.data import Dataset, DataLoader


def dna_to_one_hot(dna_sequence: str) -> np.ndarray:
    """Convert DNA into a one-hot encoded format with channel ordering ACGT.

    Args:
        dna_sequence: DNA sequence string containing A, C, G, T, N

    Returns:
        One-hot encoded array of shape (sequence_length, 4)
    """
    base_to_one_hot = {
        "A": (1, 0, 0, 0),
        "C": (0, 1, 0, 0),
        "G": (0, 0, 1, 0),
        "T": (0, 0, 0, 1),
        "N": (1, 1, 1, 1),  # N represents any unknown or ambiguous base
    }
    one_hot_encoded = np.array([base_to_one_hot[base] for base in dna_sequence])
    return one_hot_encoded.astype(np.float32)


def one_hot_to_dna(one_hot_encoded: np.ndarray) -> str:
    """Convert one-hot encoded array back to DNA sequence.

    Args:
        one_hot_encoded: Array of shape (sequence_length, 4)

    Returns:
        DNA sequence string
    """
    one_hot_to_base = {
        (1, 0, 0, 0): "A",
        (0, 1, 0, 0): "C",
        (0, 0, 1, 0): "G",
        (0, 0, 0, 1): "T",
        (1, 1, 1, 1): "N",
    }
    # Convert to tuples for lookup
    dna_sequence = "".join(
        one_hot_to_base.get(tuple(base.astype(int)), "N")
        for base in one_hot_encoded
    )
    return dna_sequence


def load_dataset(sequence_db: str) -> Dict[str, np.ndarray]:
    """Load sequences and labels from a CSV into numpy arrays.

    Args:
        sequence_db: Path to CSV file with 'sequence' and 'label' columns

    Returns:
        Dictionary with 'sequences' and 'labels' keys
    """
    df = pd.read_csv(sequence_db)
    return {
        "labels": df["label"].to_numpy()[:, None].astype(np.float32),
        "sequences": np.array([dna_to_one_hot(seq) for seq in df["sequence"]]),
    }


class DNASequenceDataset(Dataset):
    """PyTorch Dataset for DNA sequences and labels."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """Initialize dataset.

        Args:
            sequences: Array of shape (num_samples, sequence_length, 4)
            labels: Array of shape (num_samples, 1)
        """
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "sequences": self.sequences[idx],
            "labels": self.labels[idx],
        }


def load_dataset_splits(
    path: str,
    transcription_factor: str,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Load TF dataset splits (train, valid, test) as PyTorch DataLoaders.

    Args:
        path: Path to directory containing dataset CSV files
        transcription_factor: Name of transcription factor (e.g., 'CTCF')
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        Dictionary with 'train', 'valid', 'test' DataLoaders
    """
    dataset_splits = {}

    for split in ["train", "valid", "test"]:
        file_path = f"{path}/{transcription_factor}_{split}_sequences.csv"
        dataset = load_dataset(file_path)

        pytorch_dataset = DNASequenceDataset(
            dataset["sequences"],
            dataset["labels"]
        )

        dataloader = DataLoader(
            pytorch_dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        dataset_splits[split] = dataloader

    return dataset_splits


def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets.

    Args:
        labels: Array of binary labels

    Returns:
        Tensor of class weights [weight_class_0, weight_class_1]
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    return torch.tensor(weights, dtype=torch.float32)
