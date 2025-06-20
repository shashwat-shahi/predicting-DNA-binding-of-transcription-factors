"""Model interpretation techniques for understanding predictions."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from data_utils import one_hot_to_dna


def generate_all_mutations(sequence: np.ndarray) -> np.ndarray:
    """Generate all possible single base mutations of a one-hot DNA sequence.

    Args:
        sequence: One-hot encoded sequence of shape (seq_len, 4)

    Returns:
        Array of shape (seq_len * 4, seq_len, 4) containing all mutations
    """
    seq_len = sequence.shape[0]
    mutated_sequences = []

    for i in range(seq_len):
        # At each position, test all four bases (including the original)
        for j in range(4):
            mutated_sequence = sequence.copy()
            mutated_sequence[i] = np.zeros(4)
            mutated_sequence[i, j] = 1
            mutated_sequences.append(mutated_sequence)

    sequences = np.stack(mutated_sequences)
    return sequences.astype(np.float32)


@torch.no_grad()
def in_silico_saturation_mutagenesis(
    model: nn.Module,
    sequence: np.ndarray,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Perform in silico saturation mutagenesis (ISM).

    Systematically mutates each position to each of the 4 bases and measures
    the change in model prediction.

    Args:
        model: PyTorch model in eval mode
        sequence: One-hot encoded sequence of shape (seq_len, 4)
        device: Device to run on
        batch_size: Batch size for processing mutations

    Returns:
        ISM matrix of shape (seq_len, 4) with prediction delta values
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Get baseline prediction for original sequence
    sequence_tensor = torch.from_numpy(sequence[None, :, :]).float().to(device)
    baseline_logit = model(sequence_tensor).item()
    baseline_prob = 1 / (1 + np.exp(-baseline_logit))  # sigmoid

    # Generate all mutations
    all_mutations = generate_all_mutations(sequence)
    seq_len = sequence.shape[0]

    # Process mutations in batches
    ism_matrix = np.zeros((seq_len, 4))

    for i in range(0, len(all_mutations), batch_size):
        batch_mutations = all_mutations[i : i + batch_size]
        batch_tensor = torch.from_numpy(batch_mutations).float().to(device)

        # Get predictions
        logits = model(batch_tensor)
        logits = logits.cpu().numpy().flatten()

        # Convert to probabilities and compute delta
        probs = 1 / (1 + np.exp(-logits))
        deltas = probs - baseline_prob

        # Fill matrix
        for j, delta in enumerate(deltas):
            mutation_idx = i + j
            pos = mutation_idx // 4
            base = mutation_idx % 4
            ism_matrix[pos, base] = delta

    return ism_matrix


@torch.no_grad()
def compute_input_gradient(
    model: nn.Module,
    sequence: np.ndarray,
    device: Optional[torch.device] = None,
    return_raw: bool = False,
) -> np.ndarray:
    """Compute input gradient for a one-hot DNA sequence.

    Computes the gradient of model output with respect to input, showing
    which positions and bases are most important for predictions.

    Args:
        model: PyTorch model in eval mode
        sequence: One-hot encoded sequence of shape (seq_len, 4)
        device: Device to run on
        return_raw: If True, return raw gradients; if False, return absolute values

    Returns:
        Gradient array of shape (seq_len, 4)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Prepare sequence as variable that requires grad
    sequence_tensor = torch.from_numpy(sequence[None, :, :]).float().to(device)
    sequence_tensor.requires_grad_(True)

    # Forward pass
    logits = model(sequence_tensor)
    logits_sum = logits.sum()  # Ensure we have a scalar to compute grad of

    # Backward pass
    logits_sum.backward()

    # Extract gradient
    gradient = sequence_tensor.grad.squeeze(0).cpu().numpy()

    if return_raw:
        return gradient
    else:
        return np.abs(gradient)


@torch.no_grad()
def compute_attention_maps(
    model: nn.Module,
    sequence: np.ndarray,
    device: Optional[torch.device] = None,
) -> Optional[np.ndarray]:
    """Extract attention weights from transformer-based models.

    Args:
        model: PyTorch model
        sequence: One-hot encoded sequence of shape (seq_len, 4)
        device: Device to run on

    Returns:
        Attention map or None if model doesn't have attention
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Check if model has transformer blocks
    if not hasattr(model, "transformer_blocks") or len(model.transformer_blocks) == 0:
        return None

    # This would require hooking into attention layers
    # For now, return None as a placeholder
    return None


def mutation_scan(
    model: nn.Module,
    sequence: np.ndarray,
    mutation_type: str = "substitution",
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, str]:
    """Perform mutation scanning on a sequence.

    Args:
        model: PyTorch model
        sequence: One-hot encoded sequence
        mutation_type: Type of mutation ('substitution', 'deletion', 'insertion')
        device: Device to run on

    Returns:
        Tuple of (mutation_effects, sequence_string)
    """
    sequence_str = one_hot_to_dna(sequence)

    if mutation_type == "substitution":
        # ISM gives us substitution effects
        effects = in_silico_saturation_mutagenesis(model, sequence, device)
    elif mutation_type == "deletion":
        # Could implement deletion scanning
        raise NotImplementedError("Deletion scanning not yet implemented")
    elif mutation_type == "insertion":
        # Could implement insertion scanning
        raise NotImplementedError("Insertion scanning not yet implemented")
    else:
        raise ValueError(f"Unknown mutation type: {mutation_type}")

    return effects, sequence_str


@torch.no_grad()
def compute_position_importance(
    model: nn.Module,
    sequence: np.ndarray,
    method: str = "gradient",
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Compute position-level importance scores.

    Sums across all 4 bases for each position to get a single importance
    score per position.

    Args:
        model: PyTorch model
        sequence: One-hot encoded sequence of shape (seq_len, 4)
        method: Importance method ('gradient' or 'ism')
        device: Device to run on

    Returns:
        Position importance array of shape (seq_len,)
    """
    if method == "gradient":
        grad_matrix = compute_input_gradient(model, sequence, device, return_raw=False)
        # Average across bases for each position
        position_importance = grad_matrix.mean(axis=1)
    elif method == "ism":
        ism_matrix = in_silico_saturation_mutagenesis(model, sequence, device)
        # Take max absolute effect at each position
        position_importance = np.abs(ism_matrix).max(axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")

    return position_importance


@torch.no_grad()
def motif_discovery(
    model: nn.Module,
    sequences: np.ndarray,
    method: str = "gradient",
    device: Optional[torch.device] = None,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Discover important sequence motifs.

    Computes importance scores for a set of sequences and identifies
    the positions with highest importance.

    Args:
        model: PyTorch model
        sequences: Array of sequences of shape (num_seqs, seq_len, 4)
        method: Importance method
        device: Device to run on
        top_k: Number of top positions to return

    Returns:
        Tuple of (mean_importance_profiles, top_positions)
    """
    importance_profiles = []

    for seq in sequences:
        if method == "gradient":
            importance = compute_input_gradient(model, seq, device, return_raw=False)
        elif method == "ism":
            importance = in_silico_saturation_mutagenesis(model, seq, device)
        else:
            raise ValueError(f"Unknown method: {method}")

        importance_profiles.append(importance)

    importance_profiles = np.array(importance_profiles)
    mean_importance = importance_profiles.mean(axis=0)

    # Find top positions
    position_importance = mean_importance.mean(axis=1)
    top_positions = np.argsort(-position_importance)[:top_k]

    return mean_importance, top_positions


@torch.no_grad()
def perturbation_analysis(
    model: nn.Module,
    sequence: np.ndarray,
    window_size: int = 10,
    step: int = 1,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Analyze effect of perturbing windows of the sequence.

    Args:
        model: PyTorch model
        sequence: One-hot encoded sequence
        window_size: Size of perturbation window
        step: Step size between windows
        device: Device to run on

    Returns:
        Array of window effects
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Get baseline
    sequence_tensor = torch.from_numpy(sequence[None, :, :]).float().to(device)
    baseline_logit = model(sequence_tensor).item()

    seq_len = sequence.shape[0]
    num_windows = (seq_len - window_size) // step + 1
    window_effects = np.zeros(num_windows)

    for i, start_pos in enumerate(range(0, seq_len - window_size + 1, step)):
        # Perturb this window (replace with uniform distribution)
        perturbed_seq = sequence.copy()
        perturbed_seq[start_pos : start_pos + window_size] = 0.25  # uniform

        perturbed_tensor = torch.from_numpy(perturbed_seq[None, :, :]).float().to(device)
        perturbed_logit = model(perturbed_tensor).item()

        window_effects[i] = baseline_logit - perturbed_logit

    return window_effects


def conservation_analysis(
    sequences: np.ndarray,
    model: nn.Module,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Analyze conservation of important positions across sequences.

    Args:
        sequences: Array of sequences of shape (num_seqs, seq_len, 4)
        model: PyTorch model
        device: Device to run on

    Returns:
        Tuple of (conservation_scores, importance_scores)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = sequences.shape[1]
    conservation = np.zeros(seq_len)
    importance = np.zeros(seq_len)

    for seq in sequences:
        # Compute importance
        grad = compute_input_gradient(model, seq, device, return_raw=False)
        importance += grad.max(axis=1)

        # Compute conservation (entropy of bases at each position)
        for pos in range(seq_len):
            base_counts = seq[pos]
            if base_counts.sum() > 0:
                # Entropy calculation
                probs = base_counts / base_counts.sum()
                entropy = -(probs * np.log(probs + 1e-10)).sum()
                # Higher entropy = less conserved
                conservation[pos] += entropy

    conservation /= len(sequences)
    importance /= len(sequences)

    return conservation, importance
