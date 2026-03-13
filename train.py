"""
Gravitational Wave Detection using Transformer
Binary classification: 0 (no GW) vs 1 (GW detected)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import math

warnings.filterwarnings("ignore")

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Run: pip install wandb")

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    data_dir = r"D:\Code\g2net-gravitational-wave-detection"
    labels_path = os.path.join(data_dir, "training_labels.csv")
    
    # Data
    seq_length = 4096  # Length of each detector signal
    n_detectors = 3    # Number of detectors (Hanford, Livingston, Virgo)
    sample_rate = 2048
    
    # Model - Lightweight Transformer
    d_model = 64           # Embedding dimension (reduced for speed)
    n_heads = 4            # Number of attention heads
    n_layers = 2           # Number of transformer layers (reduced for speed)
    d_ff = 256             # Feed-forward dimension
    dropout = 0.1
    patch_size = 64        # Divide signal into patches for efficiency
    
    # Training
    batch_size = 96
    epochs = 20
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_workers = 0        # Set to 0 for Windows compatibility
    
    # Split
    test_size = 0.2
    random_seed = 42
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging
    use_wandb = WANDB_AVAILABLE
    project_name = "gravitational-wave-detection"
    run_name = "transformer-v1-bs-96"


# ============================================================================
# Dataset
# ============================================================================

def id2path(idx, data_dir):
    """Convert sample ID to file path"""
    return os.path.join(data_dir, "train", idx[0], idx[1], idx[2], f"{idx}.npy")


class GWDataset(Dataset):
    """Gravitational Wave Dataset"""
    
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['id']
        label = row['target']
        
        # Load raw signal (3 detectors x 4096 samples)
        path = id2path(sample_id, self.data_dir)
        signal = np.load(path).astype(np.float32)
        
        # Normalize each detector independently
        for i in range(signal.shape[0]):
            max_val = np.max(np.abs(signal[i])) + 1e-8
            signal[i] = signal[i] / max_val
        
        # Convert to tensor
        signal = torch.from_numpy(signal)  # (3, 4096)
        label = torch.tensor(label, dtype=torch.float32)
        
        return signal, label


# ============================================================================
# Model Components
# ============================================================================

class PatchEmbedding(nn.Module):
    """Convert signal into patches and embed them"""
    
    def __init__(self, n_detectors, seq_length, patch_size, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = seq_length // patch_size
        self.n_detectors = n_detectors
        
        # Linear projection for each patch
        self.projection = nn.Linear(patch_size * n_detectors, d_model)
        
        # Learnable position embeddings
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02
        )
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
    def forward(self, x):
        # x: (batch, n_detectors, seq_length)
        batch_size = x.shape[0]
        
        # Reshape to patches: (batch, n_patches, n_detectors * patch_size)
        x = x.unfold(2, self.patch_size, self.patch_size)  # (batch, n_det, n_patches, patch_size)
        x = x.permute(0, 2, 1, 3)  # (batch, n_patches, n_det, patch_size)
        x = x.reshape(batch_size, self.n_patches, -1)  # (batch, n_patches, n_det * patch_size)
        
        # Project patches
        x = self.projection(x)  # (batch, n_patches, d_model)
        
        # Add position embeddings
        x = x + self.position_embedding
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1 + n_patches, d_model)
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self Attention"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)
        
        return out


class FeedForward(nn.Module):
    """Position-wise Feed Forward Network"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single Transformer Encoder Block"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        
        return x


class GWTransformer(nn.Module):
    """Transformer for Gravitational Wave Classification"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            n_detectors=config.n_detectors,
            seq_length=config.seq_length,
            patch_size=config.patch_size,
            d_model=config.d_model
        )
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            
    def forward(self, x):
        # Patch embedding
        x = self.patch_embedding(x)  # (batch, 1 + n_patches, d_model)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Layer norm
        x = self.norm(x)
        
        # Use CLS token for classification
        cls_output = x[:, 0]  # (batch, d_model)
        
        # Classification
        logits = self.classifier(cls_output)  # (batch, 1)
        
        return logits.squeeze(-1)


# ============================================================================
# Training Functions
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, global_step, config):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    running_loss = 0
    log_interval = 100  # Log every 100 steps
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for batch_idx, (signals, labels) in enumerate(pbar):
        signals = signals.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(signals)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        running_loss += loss.item()
        global_step += 1
        
        # Collect predictions
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'step': global_step})
        
        # Log to wandb every 100 steps
        if global_step % log_interval == 0 and config.use_wandb:
            avg_running_loss = running_loss / log_interval
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                'step': global_step,
                'train/loss': avg_running_loss,
                'train/batch_loss': loss.item(),
                'train/learning_rate': current_lr
            }, step=global_step)
            running_loss = 0
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Metrics
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    
    return avg_loss, auc, acc, global_step


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in tqdm(dataloader, desc="Validating"):
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Metrics
    auc = roc_auc_score(all_labels, all_preds)
    preds_binary = (all_preds > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds_binary)
    precision = precision_score(all_labels, preds_binary, zero_division=0)
    recall = recall_score(all_labels, preds_binary, zero_division=0)
    f1 = f1_score(all_labels, preds_binary, zero_division=0)
    
    return avg_loss, auc, acc, precision, recall, f1, all_preds, all_labels


# ============================================================================
# Visualization
# ============================================================================

def plot_training_history(history, save_path='training_history.png'):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#1E1E2E')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('Loss', color='white')
    ax.set_title('Training & Validation Loss', color='white', fontweight='bold')
    ax.legend()
    ax.set_facecolor('#2D2D3D')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    
    # AUC
    ax = axes[0, 1]
    ax.plot(epochs, history['train_auc'], 'b-', label='Train AUC', linewidth=2)
    ax.plot(epochs, history['val_auc'], 'r-', label='Val AUC', linewidth=2)
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('AUC', color='white')
    ax.set_title('ROC-AUC Score', color='white', fontweight='bold')
    ax.legend()
    ax.set_facecolor('#2D2D3D')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('Accuracy', color='white')
    ax.set_title('Training & Validation Accuracy', color='white', fontweight='bold')
    ax.legend()
    ax.set_facecolor('#2D2D3D')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    
    # Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, history['lr'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('Learning Rate', color='white')
    ax.set_title('Learning Rate Schedule', color='white', fontweight='bold')
    ax.set_facecolor('#2D2D3D')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#1E1E2E', edgecolor='none')
    plt.close()
    print(f"Training history saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#1E1E2E')
    ax.set_facecolor('#2D2D3D')
    
    im = ax.imshow(cm, cmap='Blues')
    
    # Labels
    classes = ['No GW (0)', 'GW Detected (1)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, color='white')
    ax.set_yticklabels(classes, color='white')
    ax.set_xlabel('Predicted', color='white', fontsize=12)
    ax.set_ylabel('Actual', color='white', fontsize=12)
    ax.set_title('Confusion Matrix', color='white', fontweight='bold', fontsize=14)
    
    # Annotate
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha='center', va='center', 
                          color='white' if cm[i, j] > cm.max()/2 else 'black',
                          fontsize=16, fontweight='bold')
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#1E1E2E', edgecolor='none')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_pred, save_path='roc_curve.png'):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#1E1E2E')
    ax.set_facecolor('#2D2D3D')
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.3)
    
    ax.set_xlabel('False Positive Rate', color='white', fontsize=12)
    ax.set_ylabel('True Positive Rate', color='white', fontsize=12)
    ax.set_title('ROC Curve', color='white', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#1E1E2E', edgecolor='none')
    plt.close()
    print(f"ROC curve saved to {save_path}")


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    config = Config()
    
    print("=" * 60)
    print("Gravitational Wave Detection - Transformer Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Model: d_model={config.d_model}, n_heads={config.n_heads}, n_layers={config.n_layers}")
    print("=" * 60)
    
    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config={
                "d_model": config.d_model,
                "n_heads": config.n_heads,
                "n_layers": config.n_layers,
                "d_ff": config.d_ff,
                "patch_size": config.patch_size,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "epochs": config.epochs
            }
        )
    
    # Load labels
    print("\nLoading data...")
    df = pd.read_csv(config.labels_path)
    print(f"Total samples: {len(df)}")
    print(f"Class distribution: {df['target'].value_counts().to_dict()}")
    
    # Train/Val split
    train_df, val_df = train_test_split(
        df, test_size=config.test_size, 
        stratify=df['target'], 
        random_state=config.random_seed
    )
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Datasets and Dataloaders
    train_dataset = GWDataset(train_df, config.data_dir)
    val_dataset = GWDataset(val_df, config.data_dir)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, 
        shuffle=True, num_workers=config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True
    )
    
    # Model
    print("\nInitializing model...")
    model = GWTransformer(config).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss, Optimizer, Scheduler
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    total_steps = len(train_loader) * config.epochs
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader), T_mult=2)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_auc': [], 'val_auc': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }
    
    best_auc = 0
    best_model_path = 'best_model.pth'
    global_step = 0
    
    print("\nStarting training...")
    for epoch in range(config.epochs):
        # Train
        train_loss, train_auc, train_acc, global_step = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, config.device, epoch, global_step, config
        )
        
        # Validate
        val_loss, val_auc, val_acc, precision, recall, f1, val_preds, val_labels = validate(
            model, val_loader, criterion, config.device
        )
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
        print(f"  Val   - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Log epoch-level metrics to wandb
        if config.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'epoch/train_loss': train_loss,
                'epoch/val_loss': val_loss,
                'epoch/train_auc': train_auc,
                'epoch/val_auc': val_auc,
                'epoch/train_acc': train_acc,
                'epoch/val_acc': val_acc,
                'epoch/precision': precision,
                'epoch/recall': recall,
                'epoch/f1': f1,
                'epoch/learning_rate': current_lr
            }, step=global_step)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'config': config
            }, best_model_path)
            print(f"  -> New best model saved! AUC: {val_auc:.4f}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Validation AUC: {best_auc:.4f}")
    print("=" * 60)
    
    # Load best model for final evaluation
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, final_auc, final_acc, precision, recall, f1, final_preds, final_labels = validate(
        model, val_loader, criterion, config.device
    )
    
    print("\nFinal Evaluation on Validation Set:")
    print(f"  AUC: {final_auc:.4f}")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(final_labels, final_preds)
    plot_roc_curve(final_labels, final_preds)
    
    # Log final plots to wandb
    if config.use_wandb:
        wandb.log({
            "training_history": wandb.Image("training_history.png"),
            "confusion_matrix": wandb.Image("confusion_matrix.png"),
            "roc_curve": wandb.Image("roc_curve.png")
        })
        wandb.finish()
    
    print("\nAll visualizations saved!")
    print("- training_history.png")
    print("- confusion_matrix.png")
    print("- roc_curve.png")


if __name__ == "__main__":
    main()
