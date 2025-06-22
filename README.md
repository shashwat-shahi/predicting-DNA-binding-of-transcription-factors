# Transcription Factor Binding Prediction with PyTorch

A comprehensive PyTorch implementation for predicting DNA binding of transcription factors using deep learning. 

**Learning the Logic of DNA**.

## Table of Contents
1. [Overview](#overview)
2. [Biological Background](#biological-background)
3. [Machine Learning Concepts](#machine-learning-concepts)
4. [Installation](#installation)
5. [Project Structure](#project-structure)
6. [Quick Start](#quick-start)
7. [Core Components](#core-components)
8. [Model Architectures](#model-architectures)
9. [Usage Examples](#usage-examples)
10. [Interpretation Techniques](#interpretation-techniques)
11. [Performance Results](#performance-results)
12. [Troubleshooting](#troubleshooting)

## Overview

This project tackles the problem of predicting whether a DNA sequence is bound by a transcription factor (TF). Transcription factors are proteins that control gene regulation by binding to specific DNA sequences. By learning from labeled binding/non-binding sequences, deep learning models can discover the "logic" of DNA binding and make accurate predictions on new sequences.

### Key Capabilities

- **Data Processing**: One-hot encoding of DNA sequences, efficient batch loading
- **Model Architectures**: CNN, Transformer, and hybrid Conv-Transformer models
- **Training Pipeline**: Adam optimizer, cosine annealing, gradient clipping, mixed precision
- **Evaluation**: Accuracy, AUC-ROC, precision-recall curves, confusion matrices
- **Interpretation**: In Silico Saturation Mutagenesis (ISM), input gradients, position importance
- **Visualization**: Heatmaps, learning curves, ROC curves, model comparisons

### Supported Transcription Factors (10)

| TF | Difficulty | Expected AUC |
|----|-----------|--------------|
| CTCF | Easy | > 0.95 |
| ATF2 | Easy | > 0.95 |
| REST | Medium | 0.83-0.85 |
| MAX | Medium | 0.83-0.85 |
| ELK1 | Medium | 0.83-0.85 |
| SRF | Medium | 0.83-0.85 |
| ZNF24 | Hard | 0.75-0.80 |
| BACH1 | Hard | 0.75-0.80 |
| ARID3 | Hard | 0.75-0.80 |
| GABPA | Hard | 0.75-0.80 |

## Biological Background

### What is DNA?

DNA (deoxyribonucleic acid) is the molecule of life, consisting of four chemical letters or nucleotide bases:
- **A** (Adenine)
- **C** (Cytosine)
- **G** (Guanine)
- **T** (Thymine)

These bases form long sequences that carry genetic instructions. The human genome contains approximately 3.2 billion of these letters packed into 23 pairs of chromosomes.

### Gene Regulation and Transcription Factors

Only about 2% of the human genome directly codes for proteins. The remaining 98% is noncoding DNA that plays critical regulatory roles. One important class of noncoding regions contains **transcription factor binding sites** - short DNA sequences where proteins called transcription factors (TFs) attach to regulate gene activity.

Transcription factors are molecular "switches" that:
- Recognize specific DNA motifs (sequence patterns, typically 6-15 base pairs)
- Control when and where genes are expressed
- Guide cell differentiation and response to environmental signals
- Are involved in nearly every biological process

### ChIP-seq Data

The experimental data for this project comes from **ChIP-seq** (Chromatin Immunoprecipitation followed by sequencing). This technique:
1. Cross-links proteins to DNA chemically
2. Isolates DNA fragments bound by a specific TF
3. Sequences these fragments and maps them back to the genome
4. Produces a binary label: TF bound (1) or not bound (0) at each location

For this project, ChIP-seq data is simplified to binary classification: does the TF bind to this 200 bp DNA sequence or not?

## Machine Learning Concepts

### Convolutional Neural Networks (CNNs)

CNNs are powerful for learning patterns in structured data like DNA sequences. Key components:

- **Convolutional Layers**: Small learnable filters slide across the sequence, detecting local patterns
- **Pooling Layers**: Reduce dimensionality by keeping the strongest signals
- **Batch Normalization**: Stabilize training and speed up learning
- **Fully Connected Layers**: Make final predictions from learned features

**Why CNNs work for DNA:**
- DNA motifs (binding sites) are local patterns that appear anywhere in the sequence
- CNNs are translation equivariant: they recognize patterns regardless of position
- Lower layers learn simple features (GC-rich regions), higher layers learn complex patterns (motifs)

### Transformers and Self-Attention

Transformers address a key CNN limitation: they can model **long-range dependencies**.

**Self-Attention Mechanism:**
- Each position in the sequence attends to every other position
- Learns how relevant distant bases are to each other
- Produces context-aware embeddings

**Why Transformers help for DNA:**
- Some regulatory elements are thousands of bases apart
- Distant motifs can cooperatively regulate genes
- Attention weights provide interpretable focus patterns

### Model Interpretation

Understanding how models make predictions is crucial for biology. This project implements:

#### In Silico Saturation Mutagenesis (ISM)
- Systematically test all 4 bases at each position
- Generate 200 × 4 = 800 variants per sequence
- Measure how each mutation affects binding probability
- Produces a saliency map showing which positions/bases are important
- **Pros**: Direct, intuitive, biologically meaningful
- **Cons**: Computationally expensive (800 forward passes)

#### Input Gradients
- Compute ∂output/∂input via backpropagation
- Gradient magnitude indicates sensitivity to small changes
- Much faster than ISM (single backward pass)
- **Pros**: Efficient, continuous importance scores
- **Cons**: Can be noisy, less precise than ISM

#### Position Importance
- Aggregate importance across all 4 bases per position
- Identify critical sequence regions at a glance

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, scikit-learn, matplotlib, seaborn, tqdm

### Setup

```bash
# Clone or download the project
cd transcription-factors

# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'GPU: {torch.cuda.is_available()}')"
```

## Project Structure

```
transcription-factors/
├── data_utils.py              # Data loading and preprocessing
├── models.py                  # Neural network architectures
├── training.py                # Training loops and evaluation
├── interpretation.py          # ISM, gradients, analysis
├── visualization.py           # Plotting and visualization
├── info.md                    # Original specification and biological background
├── README.md                  # This file
└── requirements.txt           # Dependencies
```

### Core Modules Description

**data_utils.py** (146 lines)
- `dna_to_one_hot()` - Convert DNA sequences to one-hot encoding (200 × 4 tensors)
- `one_hot_to_dna()` - Reverse operation
- `load_dataset()` - Load CSV files with sequences and labels
- `DNASequenceDataset` - PyTorch Dataset wrapper
- `load_dataset_splits()` - Create train/valid/test DataLoaders
- `get_class_weights()` - Compute weights for imbalanced data

**models.py** (477 lines)
- `ConvModel` - Basic CNN (2 conv + 2 dense layers)
- `ConvModelV2` - Enhanced CNN with batch/layer normalization
- `ConvBlock` - Reusable convolutional block
- `MLPBlock` - Dense layer with activation and dropout
- `TransformerBlock` - Multi-head self-attention + feedforward
- `ConvTransformerModel` - Configurable combination of all components

**training.py** (438 lines)
- `MetricsLogger` - Track training metrics
- `calculate_loss()` - Compute BCE loss
- `compute_metrics()` - Calculate accuracy, AUC
- `train_step()` - Single training iteration
- `eval_step()` - Validation step
- `train()` - Full training loop with scheduling, checkpointing
- `test_model()` - Test evaluation
- Checkpoint save/load utilities

**interpretation.py** (374 lines)
- `in_silico_saturation_mutagenesis()` - Generate ISM saliency maps
- `compute_input_gradient()` - Gradient-based importance
- `compute_position_importance()` - Aggregated position importance
- `compute_attention_maps()` - Extract transformer attention
- `mutation_scan()` - Flexible mutation analysis
- `motif_discovery()` - Identify important motifs
- `perturbation_analysis()` - Window-based analysis
- `conservation_analysis()` - Cross-sequence analysis

**visualization.py** (414 lines)
- `plot_binding_site()` - Heatmaps with bases on y-axis
- `plot_10_gradients()` - Grid of multiple sequences
- `plot_learning()` - Training curves (loss, accuracy, AUC)
- `plot_roc_curve()` - ROC curve with AUC
- `plot_precision_recall_curve()` - PR curves
- `plot_position_importance()` - Bar charts
- `plot_mutations_effect()` - Combined effect plots
- `plot_model_comparison()` - Model performance comparison
- `plot_confusion_matrix()` - Confusion matrix heatmap

## Quick Start

### 1. Prepare Your Data

Create CSV files with columns: `sequence`, `label`, `transcription_factor`, `subset`

```csv
sequence,label,transcription_factor,subset
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT,1,CTCF,train
CATCAACACTCGTGCGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT,0,CTCF,train
```

File structure:
```
dna/datasets/
├── CTCF_train_sequences.csv     (60K rows)
├── CTCF_valid_sequences.csv     (10K rows)
├── CTCF_test_sequences.csv      (10K rows)
├── ATF2_train_sequences.csv
├── ATF2_valid_sequences.csv
├── ATF2_test_sequences.csv
... (repeat for all 10 TFs)
```

### 2. Train a Model

```python
import torch
from models import ConvModel
from data_utils import load_dataset_splits
from training import train

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
dataset_splits = load_dataset_splits(
    path="./dna/datasets",
    transcription_factor="CTCF",
    batch_size=32,
)

# Create model
model = ConvModel(
    conv_filters=64,
    kernel_size=10,
    dense_units=128,
    dropout_rate=0.2,
)

# Train
model, metrics = train(
    model=model,
    dataset_splits=dataset_splits,
    num_epochs=10,
    learning_rate=0.001,
    device=device,
    eval_every=100,
    save_best=True,
    checkpoint_dir="./checkpoints",
)

# Save metrics
metrics_df = metrics.to_df()
metrics_df.to_csv("training_metrics.csv", index=False)
```

### 3. Evaluate on Test Set

```python
from training import test_model

test_metrics = test_model(model, dataset_splits["test"], device=device)
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test AUC: {test_metrics['auc']:.4f}")
```

### 4. Interpret Predictions

```python
from data_utils import dna_to_one_hot, one_hot_to_dna
from interpretation import (
    in_silico_saturation_mutagenesis,
    compute_input_gradient,
)
from visualization import plot_binding_site

# Load a test sequence
sequence = dna_to_one_hot("ACGTACGTACGT...")
sequence_str = one_hot_to_dna(sequence)

# Get model prediction
with torch.no_grad():
    logit = model(torch.from_numpy(sequence[None, :, :]).float().to(device))
    prob = 1 / (1 + np.exp(-logit.item()))

print(f"Binding probability: {prob:.4f}")

# Compute ISM
ism_matrix = in_silico_saturation_mutagenesis(model, sequence, device)
plot_binding_site(
    sequence,
    ism_matrix,
    title=f"ISM (P={prob:.4f})",
    save_path="ism_plot.png",
)

# Compute gradients
gradients = compute_input_gradient(model, sequence, device)
plot_binding_site(
    sequence,
    gradients,
    title=f"Input Gradients (P={prob:.4f})",
    save_path="gradient_plot.png",
)
```

## Core Components

### Data Processing

**One-Hot Encoding:**
Each DNA base is converted to a 4-dimensional binary vector:
- A = [1, 0, 0, 0]
- C = [0, 1, 0, 0]
- G = [0, 0, 1, 0]
- T = [0, 0, 0, 1]

A 200 bp sequence becomes a 200 × 4 matrix.

**PyTorch DataLoaders:**
- Train: Shuffled, repeated for multiple epochs
- Valid/Test: Single batch for evaluation
- Automatic prefetching and pinning for GPU

### Training Configuration

```python
config = {
    # Data
    "batch_size": 32,
    "sequence_length": 200,  # Fixed
    "num_bases": 4,          # ACGT

    # Training
    "num_epochs": 10,
    "learning_rate": 0.001,
    "optimizer": "adam",

    # Model
    "conv_filters": 64,
    "kernel_size": 10,
    "dense_units": 128,
    "dropout_rate": 0.2,

    # Regularization
    "gradient_clip": 1.0,
    "weight_decay": 0.0,

    # Advanced
    "mixed_precision": False,
    "use_batch_norm": True,
}
```

## Model Architectures

### ConvModel (Basic CNN)

Input: (batch_size, 200, 4) one-hot encoded sequences

```
Conv1D(4→64, kernel=10, padding='SAME')
  ↓
BatchNorm → GELU → MaxPool(2)  → shape: (batch, 100, 64)
  ↓
Conv1D(64→64, kernel=10, padding='SAME')
  ↓
BatchNorm → GELU → MaxPool(2)  → shape: (batch, 50, 64)
  ↓
Flatten → (batch, 3200)
  ↓
Dense(3200→128) → GELU → Dropout(0.2)
  ↓
Dense(128→64) → GELU → Dropout(0.2)
  ↓
Dense(64→1) → Logits
  ↓
Output: (batch_size, 1)
```

Total parameters: ~350K

**Why this architecture:**
- 2 conv layers capture local motifs efficiently
- Kernel size 10 matches typical TF motif lengths (6-15 bp)
- MaxPooling reduces computation while preserving important signals
- 2 dense layers provide sufficient capacity for binary classification

### ConvModelV2 (Enhanced CNN)

Same as ConvModel but adds:
- Additional BatchNorm or LayerNorm options
- Configurable dropout at multiple points
- Better training stability on harder problems

### ConvTransformerModel (Hybrid Architecture)

Combines strengths of CNNs and Transformers:

```
Input
  ↓
[Conv Blocks]      → Extract local features
  ↓
[Transformer Blocks] → Model long-range interactions
  ↓
[MLP Blocks]       → Make predictions
  ↓
Output
```

**Configurable parameters:**
- `num_conv_blocks`: 1-4 (typically 2)
- `num_transformer_blocks`: 0-4 (0 = no transformers, 2 = full hybrid)
- `num_transformer_heads`: 4-16 (default 8)
- `num_mlp_blocks`: 1-3 (typically 2)

**When to use:**
- Conv only: Fast training, good for easy TFs (CTCF, ATF2)
- Hybrid: Best overall performance, captures both local and global patterns
- Transformer only: Not recommended, lacks local feature extraction

### TransformerBlock Details

Each block contains:

```
Input
  ↓
LayerNorm
  ↓
MultiHeadAttention(num_heads=8)
  ↓
Add (residual connection)
  ↓
LayerNorm
  ↓
MLP(d_model → d_model*2 → d_model)
  ↓
Add (residual connection)
  ↓
Output
```

**Multihead Attention:**
- Each head learns different aspects of dependencies
- 8 heads: Each attends to 64-dim subspaces
- Allows parallel attention on different patterns

## Usage Examples

### Example 1: Train CTCF Model

```python
import torch
import numpy as np
from models import ConvModel
from data_utils import load_dataset_splits
from training import train, test_model
from visualization import plot_learning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
dataset_splits = load_dataset_splits(
    path="./dna/datasets",
    transcription_factor="CTCF",
    batch_size=32,
    num_workers=4,
)

# Create model
model = ConvModel(conv_filters=64, kernel_size=10, dense_units=128)

# Train
model, metrics = train(
    model=model,
    dataset_splits=dataset_splits,
    num_epochs=10,
    learning_rate=0.001,
    device=device,
    eval_every=100,
    save_best=True,
    checkpoint_dir="./checkpoints/ctcf",
)

# Evaluate
test_metrics = test_model(model, dataset_splits["test"], device=device)
print(f"Test AUC: {test_metrics['auc']:.4f}")

# Visualize
plot_learning(metrics.export(), metric_name="loss", save_path="loss.png")
plot_learning(metrics.export(), metric_name="auc", save_path="auc.png")
```

### Example 2: Train Multiple TFs and Compare

```python
from models import ConvModel
import pandas as pd

transcription_factors = [
    "ARID3", "ATF2", "BACH1", "CTCF", "ELK1",
    "GABPA", "MAX", "REST", "SRF", "ZNF24",
]

results = []

for tf in transcription_factors:
    print(f"\nTraining {tf}...")

    dataset_splits = load_dataset_splits(
        "./dna/datasets", tf, batch_size=32
    )

    model = ConvModel()
    model, metrics = train(
        model, dataset_splits, num_epochs=10,
        device=device, save_best=True,
        checkpoint_dir=f"./checkpoints/{tf}",
    )

    test_metrics = test_model(model, dataset_splits["test"], device)
    results.append({
        "tf": tf,
        "accuracy": test_metrics["accuracy"],
        "auc": test_metrics["auc"],
    })

# Summary
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv("multi_tf_results.csv", index=False)
```

### Example 3: Model Interpretation with ISM and Gradients

```python
from interpretation import (
    in_silico_saturation_mutagenesis,
    compute_input_gradient,
    compute_position_importance,
)
from visualization import (
    plot_binding_site,
    plot_position_importance,
    plot_mutations_effect,
)

# Load test sequence
test_data = load_dataset("./dna/datasets/CTCF_test_sequences.csv")
sequence = test_data["sequences"][0]  # First test sequence
label = test_data["labels"][0]

# Get baseline prediction
with torch.no_grad():
    logit = model(torch.from_numpy(sequence[None, :, :]).float().to(device))
    prob = 1 / (1 + np.exp(-logit.item()))

print(f"Sequence label: {int(label[0])}, Prediction: {prob:.4f}")

# Compute ISM (mutation effects)
ism_matrix = in_silico_saturation_mutagenesis(model, sequence, device)
plot_mutations_effect(
    ism_matrix,
    baseline_prob=prob,
    save_path="ism.png",
)

# Compute input gradients (importance)
gradients = compute_input_gradient(model, sequence, device)
plot_binding_site(
    sequence,
    gradients,
    title="Input Gradients",
    save_path="gradients.png",
)

# Position importance
pos_importance = compute_position_importance(model, sequence, "gradient", device)
plot_position_importance(
    pos_importance,
    title="Position Importance",
    save_path="position_importance.png",
)
```

### Example 4: Using Transformer Models

```python
from models import ConvTransformerModel

# Create hybrid CNN-Transformer model
model = ConvTransformerModel(
    num_conv_blocks=2,           # 2 conv blocks for local features
    num_transformer_blocks=2,    # 2 transformer blocks for long-range
    num_transformer_heads=8,     # 8 attention heads
    num_mlp_blocks=2,            # 2 dense layers for final prediction
    conv_filters=64,
    dense_units=128,
)

# Train as usual
model, metrics = train(
    model, dataset_splits, num_epochs=10,
    device=device, eval_every=100,
)

# Compare architectures
print("ConvModel parameters:", sum(p.numel() for p in ConvModel().parameters()))
print("ConvTransformerModel parameters:", sum(p.numel() for p in model.parameters()))
```

### Example 5: Custom Metrics and Analysis

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Get predictions
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in dataset_splits["test"]:
        logits = model(batch["sequences"].to(device))
        preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(batch["labels"].numpy().flatten())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Compute custom metrics
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
```

## Interpretation Techniques

### In Silico Saturation Mutagenesis (ISM)

**What it does:**
For each position in the sequence, test all 4 possible bases and measure how binding probability changes.

**How it works:**
1. Get baseline prediction for original sequence: P(bind)
2. For each position i (0-199):
   - For each base b (A, C, G, T):
     - Create mutant with position i = base b
     - Get prediction: P_mut
     - Compute delta: ΔP = P_mut - P(bind)
3. Result: (200, 4) matrix of mutation effects

**Interpretation:**
- Red cells: Mutations increase binding probability
- Blue cells: Mutations decrease binding probability
- Strong patterns indicate important motifs
- Concentrated peaks = critical binding site

**Computational cost:**
- 200 positions × 4 bases = 800 forward passes
- ~2-5 seconds per sequence
- More expensive but highly informative

**Example code:**
```python
ism_matrix = in_silico_saturation_mutagenesis(model, sequence, device)
# ism_matrix shape: (200, 4)
# ism_matrix[i, j] = change in probability when position i is mutated to base j
```

### Input Gradients

**What it does:**
Computes ∂(output)/∂(input) to find which input features most affect predictions.

**How it works:**
1. Treat one-hot vectors as continuous (e.g., [0.9, 0.1, 0, 0])
2. Forward pass to get logit
3. Backward pass to compute gradients
4. Large gradient = sensitive feature

**Interpretation:**
- Magnitude indicates importance
- Sign indicates direction (increase/decrease probability)
- Faster than ISM but can be noisier

**Computational cost:**
- Single backward pass
- ~0.1 seconds per sequence
- Fast saliency map approximation

**Example code:**
```python
gradients = compute_input_gradient(model, sequence, device)
# gradients shape: (200, 4)
# gradients[i, j] = ∂output/∂input at position i, base j
```

### Position Importance

**What it does:**
Aggregates importance across all bases at each position.

**Aggregation methods:**
- Mean: `importance[i] = mean(|gradients[i, :]|)`
- Max: `importance[i] = max(|gradients[i, :]|)`

**Use case:**
Quick identification of important regions without looking at individual bases.

### Advanced Analysis

**Motif Discovery:**
Identify recurring patterns in important positions across multiple sequences.

**Perturbation Analysis:**
Measure effect of replacing windows of sequence with uniform distribution.

**Conservation Analysis:**
Compare importance across sequences to identify consensus regions.

## Performance Results

### Expected Accuracy by TF Difficulty

**Easy TFs (AUC > 0.95):**
- CTCF: Strong, well-characterized motif
- ATF2: Predictable binding patterns

**Medium TFs (AUC 0.83-0.85):**
- REST, MAX, ELK1, SRF: Moderate complexity

**Hard TFs (AUC 0.75-0.80):**
- ZNF24, BACH1, ARID3, GABPA: Complex binding patterns

### Training Time

On NVIDIA GPU (e.g., V100, A100):
- Single epoch: ~2-4 minutes
- Full training (10 epochs): ~20-40 minutes
- Test evaluation: ~1-2 minutes

On CPU:
- Single epoch: ~10-20 minutes
- Full training (10 epochs): ~100-200 minutes

### Model Sizes

- ConvModel: ~350K parameters
- ConvModelV2: ~350K parameters
- ConvTransformerModel: ~450K-600K parameters

Inference speed: <10ms per sequence on GPU

### ISM Speed

- Per sequence: 2-5 seconds
- Batch of 100: ~200-500 seconds
- Batch of 1000: ~2000-5000 seconds

## Troubleshooting

### Installation Issues

**Issue: "ModuleNotFoundError: No module named 'torch'"**
```bash
pip install torch torchvision torchaudio
# Or specific version: pip install torch==2.0.0
```

**Issue: ImportError for other packages**
```bash
pip install -r requirements.txt --upgrade
```

### GPU Issues

**Check GPU availability:**
```python
import torch
print(torch.cuda.is_available())        # Should be True
print(torch.cuda.get_device_name(0))    # Should show GPU name
print(torch.cuda.get_device_properties(0))  # Show GPU specs
```

**Force CPU if GPU issues persist:**
```python
device = torch.device("cpu")
```

### Memory Issues

**Out of Memory (OOM) Error:**
```python
# Reduce batch size
batch_size = 16  # from 32

# Or reduce model size
model = ConvModel(conv_filters=32)  # from 64

# Or enable mixed precision
model, metrics = train(
    ...,
    use_mixed_precision=True,
)
```

**Monitor memory usage:**
```bash
nvidia-smi                    # On GPU
htop                         # On CPU
```

### Training Issues

**Poor convergence:**
1. Check data loading (verify sequences are 200 bp, labels are binary)
2. Increase learning rate if loss plateaus early
3. Decrease learning rate if loss oscillates
4. Train longer (more epochs)

**Overfitting (train loss low, valid loss high):**
1. Increase dropout_rate (0.2 → 0.3 or 0.4)
2. Add L2 weight decay
3. Use early stopping

**Underfitting (both train and valid loss high):**
1. Increase model capacity (more filters, more layers)
2. Increase training time (more epochs)
3. Try hybrid CNN-Transformer model

**NaN loss:**
1. Reduce learning rate
2. Enable gradient clipping (already done)
3. Check data for invalid values

### Data Issues

**Error loading CSV files:**
```python
# Check file exists
import os
assert os.path.exists("./dna/datasets/CTCF_train_sequences.csv")

# Check CSV format
import pandas as pd
df = pd.read_csv("./dna/datasets/CTCF_train_sequences.csv")
print(df.head())
print(df.info())
```

**Sequence validation:**
```python
from data_utils import dna_to_one_hot

# Check sequence length
seq = "ACGTACGT...ACG"  # Should be 200 bp
assert len(seq) == 200

# Check valid bases
assert all(b in "ACGTN" for b in seq)

# Encode and verify
one_hot = dna_to_one_hot(seq)
assert one_hot.shape == (200, 4)
```

## Advanced Topics

### Custom Models

Create your own architecture:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size=4, seq_length=200):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=8, padding=4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=6, padding=3)
        self.fc1 = nn.Linear(64 * 50, 128)  # After 2x MaxPool
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.GELU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        # x shape: (batch, 200, 4)
        x = x.transpose(1, 2)  # → (batch, 4, 200)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### Custom Loss Functions

For class imbalance:

```python
import torch.nn as nn

# Weighted BCE
pos_weight = torch.tensor([num_negatives / num_positives])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Or focal loss for hard examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        focal_weight = (1 - probs).pow(self.gamma) * self.alpha
        loss = focal_weight * ce
        return loss.mean()
```

### Data Augmentation

For better generalization:

```python
class DNADataset(Dataset):
    def __init__(self, sequences, labels, augment=False):
        self.sequences = sequences
        self.labels = labels
        self.augment = augment

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()

        if self.augment:
            # Random shuffle within windows
            if np.random.rand() < 0.5:
                window_size = 10
                pos = np.random.randint(0, 190)
                window = seq[pos:pos+window_size]
                np.random.shuffle(window)
                seq[pos:pos+window_size] = window

            # Random base swaps
            if np.random.rand() < 0.3:
                pos = np.random.randint(0, 200)
                base = np.random.randint(0, 4)
                seq[pos] = 0
                seq[pos, base] = 1

        return {
            "sequences": torch.from_numpy(seq).float(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
```

## References

### Core Biological Concepts
- Watson & Crick (1953): Discovery of DNA structure
- ENCODE Consortium: Genome annotation and regulation
- Various ChIP-seq studies: TF binding characterization

### Deep Learning for Biology
- DeepBind (Alipanahi et al., 2015): Early CNN for TF binding
- DanQ (Quang & Xie, 2016): CNN-RNN hybrid model
- BPNet (Avsec et al., 2021): Interpretable base pair network

### Deep Learning Methods
- LeCun et al. (1998): CNN fundamentals
- Vaswani et al. (2017): "Attention is All You Need" - Transformer architecture
- Simonyan et al. (2013): Visualizing CNNs via gradients
- Zeiler & Fergus (2013): Deconvolutional visualization

### Interpretation Methods
- Simonyan et al. (2013): Input gradients/saliency maps
- Zeiler & Fergus (2013): Deconvolutional networks
- Ribeiro et al. (2016): LIME for interpretability
- Sundararajan et al. (2017): Integrated gradients

### Relevant Tools
- PyTorch: https://pytorch.org/
- Scikit-learn: https://scikit-learn.org/
- Matplotlib/Seaborn: Plotting and visualization
