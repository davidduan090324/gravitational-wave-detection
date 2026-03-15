# Gravitational Wave Transformer - Model Architecture

## Overview

This document describes the architecture of the Transformer model used for gravitational wave signal classification.

## Model Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GRAVITATIONAL WAVE TRANSFORMER                           │
│                      Binary Classification Model                            │
└─────────────────────────────────────────────────────────────────────────────┘

                              INPUT
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Raw Signal Input                                    │
│                    Shape: (batch, 3, 4096)                                  │
│         3 detectors: LIGO Hanford, LIGO Livingston, Virgo                   │
│                    4096 samples per detector                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PATCH EMBEDDING LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  1. Patch Division                                                  │    │
│  │     - Divide each 4096-sample signal into 64-sample patches         │    │
│  │     - Results in 64 patches per sample                              │    │
│  │     - Shape: (batch, 64, 3*64) = (batch, 64, 192)                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                │                                            │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  2. Linear Projection                                               │    │
│  │     - Project patches to d_model dimension                          │    │
│  │     - Linear(192, 64)                                               │    │
│  │     - Shape: (batch, 64, 64)                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                │                                            │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  3. Add Position Embeddings                                         │    │
│  │     - Learnable position embeddings                                 │    │
│  │     - Shape: (1, 64, 64) broadcast to (batch, 64, 64)               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                │                                            │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  4. Prepend CLS Token                                               │    │
│  │     - Learnable [CLS] token for classification                      │    │
│  │     - Final shape: (batch, 65, 64)                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER ENCODER (×2 layers)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    TRANSFORMER BLOCK                                  │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Layer Norm                                                     │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                              │                                        │  │
│  │                              ▼                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Multi-Head Self-Attention (4 heads)                            │  │  │
│  │  │  ┌───────────┬───────────┬───────────┬───────────┐              │  │  │
│  │  │  │  Head 1   │  Head 2   │  Head 3   │  Head 4   │              │  │  │
│  │  │  │  d_k=16   │  d_k=16   │  d_k=16   │  d_k=16   │              │  │  │
│  │  │  └───────────┴───────────┴───────────┴───────────┘              │  │  │
│  │  │                      │                                          │  │  │
│  │  │                      ▼                                          │  │  │
│  │  │              Concatenate + Linear(64, 64)                       │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                              │                                        │  │
│  │                    ┌─────────┴─────────┐                              │  │
│  │                    │    + Residual     │                              │  │
│  │                    └─────────┬─────────┘                              │  │
│  │                              │                                        │  │
│  │                              ▼                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Layer Norm                                                     │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                              │                                        │  │
│  │                              ▼                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Feed-Forward Network                                           │  │  │
│  │  │  ┌─────────────────────────────────────────────────────────┐    │  │  │
│  │  │  │  Linear(64, 256) → GELU → Dropout → Linear(256, 64)     │    │  │  │
│  │  │  └─────────────────────────────────────────────────────────┘    │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                              │                                        │  │
│  │                    ┌─────────┴─────────┐                              │  │
│  │                    │    + Residual     │                              │  │
│  │                    └─────────┬─────────┘                              │  │
│  │                              │                                        │  │
│  └──────────────────────────────┼────────────────────────────────────────┘  │
│                                 │                                           │
│                                 ▼                                           │
│                          [Repeat ×2]                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FINAL LAYER NORM                                    │
│                        Shape: (batch, 65, 64)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EXTRACT CLS TOKEN                                     │
│                      Take first token only                                  │
│                        Shape: (batch, 64)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CLASSIFICATION HEAD                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Linear(64, 32) → GELU → Dropout(0.1) → Linear(32, 1)               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                        Shape: (batch, 1)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT                                           │
│                      Logits: (batch,)                                       │
│              Apply Sigmoid for probability [0, 1]                           │
│                    0 = No GW, 1 = GW Detected                               │
└─────────────────────────────────────────────────────────────────────────────┘


## Data Flow Summary

```
Input: (batch, 3, 4096)
    │
    ├── Patch Division ──────────────────► (batch, 64, 192)
    │
    ├── Linear Projection ───────────────► (batch, 64, 64)
    │
    ├── + Position Embedding ────────────► (batch, 64, 64)
    │
    ├── Prepend CLS Token ───────────────► (batch, 65, 64)
    │
    ├── Transformer Block ×2 ────────────► (batch, 65, 64)
    │
    ├── Layer Norm ──────────────────────► (batch, 65, 64)
    │
    ├── Extract CLS Token ───────────────► (batch, 64)
    │
    └── Classification Head ─────────────► (batch, 1) ──► Sigmoid ──► [0, 1]
```


## Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 64 | Embedding dimension |
| `n_heads` | 4 | Number of attention heads |
| `n_layers` | 2 | Number of transformer blocks |
| `d_ff` | 256 | Feed-forward hidden dimension |
| `patch_size` | 64 | Size of each signal patch |
| `n_patches` | 64 | Number of patches (4096/64) |
| `dropout` | 0.1 | Dropout rate |
| `seq_length` | 4096 | Input signal length |
| `n_detectors` | 3 | Number of detectors |


## Parameter Count

```
Total Parameters: ~50,000
├── Patch Embedding
│   ├── Projection: 192 × 64 = 12,288
│   ├── Position Embedding: 64 × 64 = 4,096
│   └── CLS Token: 64
│
├── Transformer Blocks (×2)
│   ├── Multi-Head Attention
│   │   ├── W_q, W_k, W_v: 3 × 64 × 64 = 12,288 (×2)
│   │   └── W_o: 64 × 64 = 4,096 (×2)
│   │
│   ├── Feed-Forward
│   │   ├── Linear1: 64 × 256 = 16,384 (×2)
│   │   └── Linear2: 256 × 64 = 16,384 (×2)
│   │
│   └── Layer Norms: 4 × 64 × 2 = 512 (×2)
│
└── Classification Head
    ├── Linear1: 64 × 32 = 2,048
    └── Linear2: 32 × 1 = 32
```


## Loss Function: Focal Loss

Focal Loss is used to handle class imbalance:

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where:
- $\alpha = 0.25$ (weighting factor)
- $\gamma = 2.0$ (focusing parameter)
- $p_t$ is the model's estimated probability for the correct class


## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
     │   Raw Data   │ ──►  │  80% Train   │ ──►  │  DataLoader  │
     │ (.npy files) │      │  20% Val     │      │  batch=32    │
     └──────────────┘      └──────────────┘      └──────────────┘
                                                        │
                                                        ▼
     ┌──────────────────────────────────────────────────────────────┐
     │                     TRAINING LOOP                            │
     │  ┌────────────────────────────────────────────────────────┐  │
     │  │  For each epoch:                                       │  │
     │  │    1. Forward pass through Transformer                 │  │
     │  │    2. Compute Focal Loss                               │  │
     │  │    3. Backward pass (gradients)                        │  │
     │  │    4. Optimizer step (AdamW)                           │  │
     │  │    5. Learning rate schedule (CosineAnnealingWarmRestarts)│ │
     │  │    6. Validation evaluation                            │  │
     │  │    7. Save best model (by AUC)                         │  │
     │  └────────────────────────────────────────────────────────┘  │
     └──────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
     ┌──────────────────────────────────────────────────────────────┐
     │                      EVALUATION                              │
     │  Metrics: AUC, Accuracy, Precision, Recall, F1              │
     │  Visualizations: Training curves, ROC curve, Confusion Matrix│
     └──────────────────────────────────────────────────────────────┘


## Key Design Choices

### 1. Patch-based Processing
Instead of processing the full 4096-length signal directly, we divide it into 64-sample patches. This reduces computational complexity from O(n²) to O((n/p)²) where p is patch size.

### 2. Lightweight Architecture
- Only 2 transformer layers (vs 12+ in BERT/ViT)
- Small embedding dimension (64 vs 768)
- Enables training on consumer GPUs in reasonable time

### 3. CLS Token for Classification
Following BERT/ViT convention, a learnable [CLS] token is prepended to the sequence. The final representation of this token is used for classification.

### 4. Focal Loss
Addresses potential class imbalance in gravitational wave detection datasets, where positive samples (actual GW events) may be rare.

### 5. Cosine Annealing with Warm Restarts
Learning rate scheduler that periodically resets the learning rate, helping escape local minima and improve convergence.


## Output Files

After training, the following files are generated:

| File | Description |
|------|-------------|
| `best_model.pth` | Best model checkpoint (by validation AUC) |
| `training_history.png` | Training/validation loss, AUC, accuracy curves |
| `confusion_matrix.png` | Confusion matrix visualization |
| `roc_curve.png` | ROC curve with AUC score |


## Usage

```python
# Train the model
python train.py

# The model will:
# 1. Load training_labels.csv
# 2. Split data 80/20 train/val
# 3. Train for 20 epochs
# 4. Save best model and visualizations
# 5. Log to wandb (if available)
```


## Hardware Requirements

- **Minimum**: 8GB RAM, GTX 1060 6GB (or CPU)
- **Recommended**: 16GB RAM, RTX 3060+ 8GB
- **Training Time**: ~15-30 minutes on GPU, ~2-4 hours on CPU (20 epochs)
