"""
Gravitational Wave Detection using CQT Spectrogram + Deep Learning Models
Supports: EfficientNet, ResNet, Vision Transformer (ViT), SimpleCNN
Binary classification: 0 (no GW) vs 1 (GW detected)

Usage:
    python train.py --exp_name "baseline_v1"
    python train.py --exp_name "efficientnet_b0_lr1e4" --epochs 20 --lr 1e-4
    python train.py --exp_name "transformer_test" --model transformer
    python train.py --exp_name "resnet50_test" --model resnet50
"""

import os
import time
import argparse
from datetime import datetime
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings("ignore")

# CQT Transform
from nnAudio.Spectrogram import CQT1992v2

# Import models
from models import get_model

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Run: pip install wandb")


# ============================================================================
# Argument Parser
# ============================================================================

AVAILABLE_MODELS = [
    # EfficientNet family
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    # ResNet family
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    # Transformer family
    'transformer', 'transformer-small', 'transformer-base', 'transformer-large',
    'transformer_small', 'transformer_base', 'transformer_large',
    # Simple CNN
    'simple_cnn'
]

MODEL_ALIASES = {
    'transformer_small': 'transformer-small',
    'transformer_base': 'transformer-base',
    'transformer_large': 'transformer-large',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Gravitational Wave Detection Training')
    
    # Required argument
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment name (required). Used for result directory naming.')
    
    # Optional arguments
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model', type=str, default='efficientnet-b0', 
                        choices=AVAILABLE_MODELS,
                        help='Model architecture. Options: efficientnet-b0~b7, resnet18/34/50/101/152, '
                             'transformer/transformer-small/transformer-large, simple_cnn')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='Use pretrained backbone weights when available')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false',
                        help='Disable pretrained backbone weights')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.set_defaults(pretrained=True)
    
    return parser.parse_args()


# ============================================================================
# Configuration
# ============================================================================

class Config:
    def __init__(self, args=None):
        # Paths
        self.data_dir = r"D:\Code\g2net-gravitational-wave-detection"
        self.labels_path = os.path.join(self.data_dir, "training_labels.csv")
        
        # Data
        self.seq_length = 4096
        self.n_detectors = 3
        self.sample_rate = 2048
        
        # CQT Parameters (from successful Kaggle solution)
        self.cqt_fmin = 20
        self.cqt_fmax = 1024
        self.cqt_hop_length = 32
        
        # Model
        raw_model_name = args.model if args else "efficientnet-b0"
        self.model_name = MODEL_ALIASES.get(raw_model_name, raw_model_name)
        self.pretrained = args.pretrained if args else True
        
        # Training
        self.batch_size = args.batch_size if args else 256
        self.epochs = args.epochs if args else 15
        self.learning_rate = args.lr if args else 1e-4
        self.weight_decay = 1e-5
        self.num_workers = 0  # Set to 0 for Windows
        
        # Split
        self.test_size = 0.2
        self.random_seed = args.seed if args else 42
        self.use_kfold = False
        self.n_folds = 5
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Experiment info
        self.exp_name = args.exp_name if args else "default"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create result directory
        self.result_dir = os.path.join("results", f"{self.timestamp}_{self.exp_name}")
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Output paths
        self.model_save_path = os.path.join(self.result_dir, "best_model.pth")
        self.history_plot_path = os.path.join(self.result_dir, "training_history.png")
        self.confusion_matrix_path = os.path.join(self.result_dir, "confusion_matrix.png")
        self.roc_curve_path = os.path.join(self.result_dir, "roc_curve.png")
        self.config_path = os.path.join(self.result_dir, "config.json")
        self.log_path = os.path.join(self.result_dir, "training.log")
        
        # Logging
        self.use_wandb = WANDB_AVAILABLE and not (args.no_wandb if args else False)
        self.project_name = "gravitational-wave-detection"
        self.run_name = f"{self.timestamp}_{self.exp_name}"
    
    def save_config(self):
        """Save configuration to JSON file"""
        config_dict = {
            "exp_name": self.exp_name,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "random_seed": self.random_seed,
            "cqt_fmin": self.cqt_fmin,
            "cqt_fmax": self.cqt_fmax,
            "cqt_hop_length": self.cqt_hop_length,
            "test_size": self.test_size,
            "device": str(self.device),
            "use_wandb": self.use_wandb,
        }
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"Config saved to {self.config_path}")


# ============================================================================
# Dataset with CQT Transform
# ============================================================================

def id2path(idx, data_dir):
    """Convert sample ID to file path"""
    return os.path.join(data_dir, "train", idx[0], idx[1], idx[2], f"{idx}.npy")


class GWDataset(Dataset):
    """Gravitational Wave Dataset with CQT Transform (Kaggle winning approach)"""
    
    def __init__(self, df, data_dir, config, is_train=True):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.config = config
        self.is_train = is_train
        
        # CQT Transform - same parameters as Kaggle solution
        self.q_transform = CQT1992v2(
            sr=config.sample_rate,
            fmin=config.cqt_fmin,
            fmax=config.cqt_fmax,
            hop_length=config.cqt_hop_length
        )
        
    def __len__(self):
        return len(self.df)
    
    def __get_qtransform(self, x):
        """Convert 3-channel signal to 3-channel CQT spectrogram"""
        image = []
        for i in range(3):
            wave = x[i]
            wave_tensor = torch.from_numpy(wave).float()
            
            # Apply CQT transform
            cqt_out = self.q_transform(wave_tensor).squeeze().numpy()
            
            # Convert to log scale (dB) - this is crucial!
            cqt_db = np.log10(cqt_out + 1e-30) * 20
            
            # Normalize to [0, 1]
            c_min, c_max = cqt_db.min(), cqt_db.max()
            if c_max - c_min > 1e-10:
                cqt_norm = (cqt_db - c_min) / (c_max - c_min)
            else:
                cqt_norm = np.zeros_like(cqt_db) + 0.5
            
            image.append(cqt_norm)
        
        # Stack as 3-channel image: (3, freq_bins, time_bins)
        image = np.stack(image, axis=0).astype(np.float32)
        
        return torch.tensor(image).float()
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['id']
        label = row['target']
        
        # Load raw signal (3 detectors x 4096 samples)
        path = id2path(sample_id, self.data_dir)
        x = np.load(path).astype(np.float32)
        
        # Convert to CQT spectrogram
        image = self.__get_qtransform(x)
        
        # Data augmentation during training
        if self.is_train:
            # Random horizontal flip (time axis)
            if np.random.random() < 0.5:
                image = torch.flip(image, dims=[2])
            
            # Random amplitude scaling
            if np.random.random() < 0.3:
                scale = np.random.uniform(0.9, 1.1)
                image = image * scale
        
        label = torch.tensor(label, dtype=torch.float)
        
        return {"X": image, "y": label}


# ============================================================================
# Metrics
# ============================================================================

class LossMeter:
    """Track average loss"""
    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, val):
        self.n += 1
        self.avg = val / self.n + (self.n - 1) / self.n * self.avg


class AccMeter:
    """Track accuracy"""
    def __init__(self):
        self.avg = 0
        self.n = 0
        
    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().astype(int)
        y_pred = (y_pred.cpu().numpy() >= 0).astype(int)  # sigmoid(0) = 0.5
        last_n = self.n
        self.n += len(y_true)
        true_count = np.sum(y_true == y_pred)
        self.avg = true_count / self.n + last_n / self.n * self.avg


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    def __init__(self, model, device, optimizer, criterion, config):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        
        self.best_valid_score = -np.inf
        self.best_valid_auc = 0
        self.global_step = 0
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_auc': [], 'val_auc': [],
            'lr': []
        }
    
    def fit(self, epochs, train_loader, valid_loader, save_path, scheduler=None):
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs}")
            print('='*60)
            
            # Train
            train_loss, train_acc, train_auc, train_time = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc, val_auc, val_time, val_preds, val_labels = self.valid_epoch(valid_loader)
            
            # Step scheduler
            if scheduler is not None:
                scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            self.history['lr'].append(current_lr)
            
            # Print metrics
            print(f"\n[Train] Loss: {train_loss:.5f}, Acc: {train_acc:.5f}, AUC: {train_auc:.5f}, Time: {train_time}s")
            print(f"[Valid] Loss: {val_loss:.5f}, Acc: {val_acc:.5f}, AUC: {val_auc:.5f}, Time: {val_time}s")
            print(f"LR: {current_lr:.6f}")
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/acc': train_acc,
                    'train/auc': train_auc,
                    'val/loss': val_loss,
                    'val/acc': val_acc,
                    'val/auc': val_auc,
                    'learning_rate': current_lr
                }, step=self.global_step)
            
            # Save best model
            if val_auc > self.best_valid_auc:
                print(f"  -> New best model! AUC improved from {self.best_valid_auc:.5f} to {val_auc:.5f}")
                self.best_valid_auc = val_auc
                self.save_model(epoch, save_path)
        
        return self.history, val_preds, val_labels
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        t = time.time()
        
        loss_meter = LossMeter()
        acc_meter = AccMeter()
        all_preds = []
        all_labels = []
        running_loss = 0
        log_interval = 100
        
        pbar = tqdm(train_loader, desc=f"Training")
        for step, batch in enumerate(pbar, 1):
            X = batch["X"].to(self.device)
            targets = batch["y"].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X)
            
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Debug: Print gradient info on first step of first epoch
            if step == 1 and epoch == 1:
                total_grad_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                print(f"\n[DEBUG] Step 1 - Loss: {loss.item():.4f}, Grad norm: {total_grad_norm:.6f}")
                print(f"[DEBUG] Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                print(f"[DEBUG] Targets: {targets[:5].cpu().numpy()}")
                print(f"[DEBUG] Outputs: {outputs[:5].detach().cpu().numpy()}")
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            loss_meter.update(loss.detach().item())
            acc_meter.update(targets, outputs.detach())
            
            # Collect predictions for AUC
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(targets.cpu().numpy())
            
            running_loss += loss.item()
            self.global_step += 1
            
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}'
            })
            
            # Log to wandb every N steps
            if self.global_step % log_interval == 0 and self.config.use_wandb:
                avg_running_loss = running_loss / log_interval
                wandb.log({
                    'step': self.global_step,
                    'train/step_loss': avg_running_loss,
                }, step=self.global_step)
                running_loss = 0
        
        # Calculate AUC
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        train_auc = roc_auc_score(all_labels, all_preds)
        
        return loss_meter.avg, acc_meter.avg, train_auc, int(time.time() - t)
    
    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        
        loss_meter = LossMeter()
        acc_meter = AccMeter()
        all_preds = []
        all_labels = []
        
        pbar = tqdm(valid_loader, desc="Validating")
        with torch.no_grad():
            for step, batch in enumerate(pbar, 1):
                X = batch["X"].to(self.device)
                targets = batch["y"].to(self.device)
                
                outputs = self.model(X)
                loss = self.criterion(outputs, targets)
                
                loss_meter.update(loss.detach().item())
                acc_meter.update(targets, outputs)
                
                # Collect predictions
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(targets.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'acc': f'{acc_meter.avg:.4f}'
                })
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        val_auc = roc_auc_score(all_labels, all_preds)
        
        return loss_meter.avg, acc_meter.avg, val_auc, int(time.time() - t), all_preds, all_labels
    
    def save_model(self, epoch, save_path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_valid_auc": self.best_valid_auc,
            "epoch": epoch,
        }, save_path)


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
    
    classes = ['No GW (0)', 'GW Detected (1)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, color='white')
    ax.set_yticklabels(classes, color='white')
    ax.set_xlabel('Predicted', color='white', fontsize=12)
    ax.set_ylabel('Actual', color='white', fontsize=12)
    ax.set_title('Confusion Matrix', color='white', fontweight='bold', fontsize=14)
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', 
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
# Logger
# ============================================================================

class Logger:
    """Simple logger that writes to both console and file"""
    def __init__(self, log_path):
        self.log_path = log_path
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


# ============================================================================
# Main
# ============================================================================

def main():
    # Parse arguments
    args = parse_args()
    
    # Create config with args
    config = Config(args)
    
    # Setup logging to file
    logger = Logger(config.log_path)
    sys.stdout = logger
    
    print("=" * 60)
    print("Gravitational Wave Detection")
    print("Model: CQT Spectrogram + Deep Learning")
    print("=" * 60)
    print(f"Experiment: {config.exp_name}")
    print(f"Timestamp: {config.timestamp}")
    print(f"Result Directory: {config.result_dir}")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Model: {config.model_name}")
    print(f"Pretrained: {config.pretrained}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"CQT: fmin={config.cqt_fmin}, fmax={config.cqt_fmax}, hop={config.cqt_hop_length}")
    print("=" * 60)
    
    # Save config
    config.save_config()
    
    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config={
                "exp_name": config.exp_name,
                "model": config.model_name,
                "pretrained": config.pretrained,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "cqt_fmin": config.cqt_fmin,
                "cqt_fmax": config.cqt_fmax,
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
    
    # Datasets
    train_dataset = GWDataset(train_df, config.data_dir, config, is_train=True)
    val_dataset = GWDataset(val_df, config.data_dir, config, is_train=False)
    
    # Debug: Check data shape and range
    print("\n[DEBUG] Checking data...")
    sample = train_dataset[0]
    print(f"  Image shape: {sample['X'].shape}")
    print(f"  Image range: [{sample['X'].min():.4f}, {sample['X'].max():.4f}]")
    print(f"  Image mean: {sample['X'].mean():.4f}, std: {sample['X'].std():.4f}")
    print(f"  Label: {sample['y']}")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Model
    print("\nInitializing model...")
    model = get_model(config.model_name, config).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Debug: Test forward pass
    print("\n[DEBUG] Testing forward pass...")
    model.eval()
    with torch.no_grad():
        test_input = sample['X'].unsqueeze(0).to(config.device)
        test_output = model(test_input)
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {test_output.shape}")
        print(f"  Output value: {test_output.item():.4f}")
        print(f"  Sigmoid output: {torch.sigmoid(test_output).item():.4f}")
    
    # Loss, Optimizer, Scheduler
    criterion = F.binary_cross_entropy_with_logits
    if config.model_name.startswith('transformer'):
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )
    
    # Trainer
    trainer = Trainer(model, config.device, optimizer, criterion, config)
    
    # Train
    print("\nStarting training...")
    history, val_preds, val_labels = trainer.fit(
        epochs=config.epochs,
        train_loader=train_loader,
        valid_loader=val_loader,
        save_path=config.model_save_path,
        scheduler=scheduler
    )
    
    # Final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Validation AUC: {trainer.best_valid_auc:.5f}")
    print("=" * 60)
    
    # Final metrics
    final_acc = accuracy_score(val_labels, (val_preds > 0.5).astype(int))
    precision = precision_score(val_labels, (val_preds > 0.5).astype(int), zero_division=0)
    recall = recall_score(val_labels, (val_preds > 0.5).astype(int), zero_division=0)
    f1 = f1_score(val_labels, (val_preds > 0.5).astype(int), zero_division=0)
    
    print("\nFinal Validation Metrics:")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Save final metrics to JSON
    metrics_path = os.path.join(config.result_dir, "metrics.json")
    metrics = {
        "best_auc": trainer.best_valid_auc,
        "final_accuracy": final_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    # Plot results (save to result directory)
    plot_training_history(history, config.history_plot_path)
    plot_confusion_matrix(val_labels, val_preds, config.confusion_matrix_path)
    plot_roc_curve(val_labels, val_preds, config.roc_curve_path)
    
    # Save training history to JSON
    history_json_path = os.path.join(config.result_dir, "history.json")
    with open(history_json_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_json_path}")
    
    # Log final plots to wandb
    if config.use_wandb:
        wandb.log({
            "training_history": wandb.Image(config.history_plot_path),
            "confusion_matrix": wandb.Image(config.confusion_matrix_path),
            "roc_curve": wandb.Image(config.roc_curve_path),
            "best_auc": trainer.best_valid_auc,
            "final_accuracy": final_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        wandb.finish()
    
    print("\n" + "=" * 60)
    print(f"All results saved to: {config.result_dir}")
    print("=" * 60)
    print("Files:")
    print(f"  - config.json")
    print(f"  - training.log")
    print(f"  - best_model.pth")
    print(f"  - metrics.json")
    print(f"  - history.json")
    print(f"  - training_history.png")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_curve.png")
    
    # Close logger
    logger.close()
    sys.stdout = logger.terminal


if __name__ == "__main__":
    main()
