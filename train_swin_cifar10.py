"""
Reduced-scale Swin Transformer training on CIFAR-10
- Smaller model (fewer layers/heads, reduced embedding dims)
- Quick iteration (reduced epochs, subset data)
- Generates: loss/accuracy curves, final metrics, training time summary, validation samples
"""

import os
import sys
import time
import json
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Add Swin-Transformer-main to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Swin-Transformer-main'))

from models.swin_transformer import SwinTransformer
from timm.loss import LabelSmoothingCrossEntropy

# =============================================================================
# CONFIGURATION - Reduced Complexity
# =============================================================================
CONFIG = {
    'img_size': 32,
    'patch_size': 4,
    'in_chans': 3,
    'num_classes': 10,
    'embed_dim': 48,
    'depths': [2, 2, 2],        # Slightly deeper: 3 stages instead of 2
    'num_heads': [4, 8, 16],    # More heads in later stages
    'window_size': 4,
    'mlp_ratio': 4,
    'qkv_bias': True,
    'drop_rate': 0.0,
    'drop_path_rate': 0.15,
    'batch_size': 128,          # Larger batch for better gradient estimates
    'learning_rate': 3e-4,      # Slightly lower for stability
    'weight_decay': 0.01,
    'label_smoothing': 0.1,
    'train_subset_pct': 0.5,    # Use 50% training data
    'val_subset_pct': 1.0,      # Use full validation set
    'seed': 42,
    'target_acc': 75.0,         # Target validation accuracy
    'max_epochs': 500,          # Cap at 500 to avoid infinite loop
    'output_dir': os.path.join(os.path.dirname(__file__), 'outputs', 'swin_cifar10_demo'),
    # --- Checkpointing ---
    'save_every_n_epochs': 10,  # Save a periodic checkpoint every N epochs
}

# =============================================================================
# SETUP
# =============================================================================
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_model(config, device):
    """Create reduced Swin Transformer"""
    model = SwinTransformer(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        in_chans=config['in_chans'],
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        depths=config['depths'],
        num_heads=config['num_heads'],
        window_size=config['window_size'],
        mlp_ratio=config['mlp_ratio'],
        qkv_bias=config['qkv_bias'],
        drop_rate=config['drop_rate'],
        drop_path_rate=config['drop_path_rate'],
    )
    return model.to(device)

# =============================================================================
# CHECKPOINTING
# =============================================================================
def save_checkpoint(state, path, tag=''):
    """Save a checkpoint dict to disk."""
    torch.save(state, path)
    print(f"  [OK] Checkpoint saved [{tag}] -> {path}")

def load_checkpoint(path, model, optimizer, scheduler, device):
    """
    Load a checkpoint and restore model/optimizer/scheduler state.
    Returns the epoch to resume from and the best val_acc seen so far.
    """
    print(f"Loading checkpoint from {path} ...")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch  = checkpoint['epoch']          # next epoch to run
    best_val_acc = checkpoint['best_val_acc']
    history      = checkpoint['history']
    print(f"  Resumed from epoch {start_epoch} | best val acc so far: {best_val_acc:.2f}%")
    return start_epoch, best_val_acc, history

# =============================================================================
# DATA
# =============================================================================
def get_data_loaders(config):
    """Download and prepare CIFAR-10 data loaders"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Download CIFAR-10
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'cifar10')
    os.makedirs(data_dir, exist_ok=True)

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=transform_train)
    val_dataset   = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_val)

    # Create subsets for quick iteration
    train_subset_size = int(len(train_dataset) * config['train_subset_pct'])
    val_subset_size   = int(len(val_dataset)   * config['val_subset_pct'])

    train_subset = Subset(train_dataset, list(range(train_subset_size)))
    val_subset   = Subset(val_dataset,   list(range(val_subset_size)))

    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_subset,   batch_size=config['batch_size'], shuffle=False, num_workers=0)

    return train_loader, val_loader, len(train_subset), len(val_subset)

# =============================================================================
# TRAINING
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, epoch, config, device):
    model.train()
    losses  = []
    correct = 0
    total   = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return np.mean(losses), 100. * correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses  = []
    correct = 0
    total   = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss    = criterion(outputs, targets)

        losses.append(loss.item())
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return np.mean(losses), 100. * correct / total

# =============================================================================
# VISUALIZATION
# =============================================================================
def save_validation_samples(model, val_loader, config, device, num_samples=16):
    """Save sample validation predictions as images"""
    model.eval()

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    samples = []
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for i in range(min(images.size(0), num_samples - len(samples))):
                samples.append({
                    'image':     images[i].cpu(),
                    'target':    targets[i].item(),
                    'predicted': predicted[i].item(),
                    'correct':   predicted[i].item() == targets[i].item()
                })
            if len(samples) >= num_samples:
                break

    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    fig.suptitle('Validation Sample Predictions', fontsize=14, fontweight='bold')

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(samples):
            ax.axis('off')
            continue

        sample = samples[idx]
        img = torch.clamp(sample['image'] * std + mean, 0, 1)

        ax.imshow(img.permute(1, 2, 0))
        ax.axis('off')

        color = 'green' if sample['correct'] else 'red'
        ax.set_title(f"T: {class_names[sample['target']]}\nP: {class_names[sample['predicted']]}",
                     color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'validation_samples.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Validation samples saved to {config['output_dir']}/validation_samples.png")

    n_correct = sum(1 for s in samples if s['correct'])
    return {'total': len(samples), 'correct': n_correct, 'accuracy': 100. * n_correct / len(samples)}

# =============================================================================
# MAIN
# =============================================================================
def main():
    config = CONFIG
    os.makedirs(config['output_dir'], exist_ok=True)
    setup_seed(config['seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Model config: embed_dim={config['embed_dim']}, depths={config['depths']}, num_heads={config['num_heads']}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, n_train, n_val = get_data_loaders(config)
    print(f"Training samples: {n_train}, Validation samples: {n_val}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model    = create_model(config, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Optimiser / scheduler ─────────────────────────────────────────────────
    criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epochs'])

    # ── Checkpoint paths ──────────────────────────────────────────────────────
    best_ckpt     = os.path.join(config['output_dir'], 'best_model.pth')
    periodic_ckpt = os.path.join(config['output_dir'], 'checkpoint_latest.pth')

    # ── Resume if a checkpoint exists ─────────────────────────────────────────
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'epoch_times': []}
    start_epoch  = 0
    best_val_acc = 0.0

    if os.path.exists(periodic_ckpt):
        ans = input(f"\nCheckpoint found at {periodic_ckpt}. Resume? [y/n]: ").strip().lower()
        if ans == 'y':
            start_epoch, best_val_acc, history = load_checkpoint(
                periodic_ckpt, model, optimizer, scheduler, device)

    # ── Training loop ─────────────────────────────────────────────────────────
    target_acc = config['target_acc']
    max_epochs = config['max_epochs']
    print(f"\nStarting training (target: {target_acc}% val acc, max {max_epochs} epochs)...")
    start_time = time.time()

    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, config, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{max_epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s | LR: {lr:.6f}")

        # ── Save best model whenever val_acc improves ──────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch':                epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc':         best_val_acc,
                'history':              history,
            }, best_ckpt, tag=f'BEST  val_acc={best_val_acc:.2f}%')

        # ── Save periodic checkpoint every N epochs ────────────────────────────
        if (epoch + 1) % config['save_every_n_epochs'] == 0:
            save_checkpoint({
                'epoch':                epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc':         best_val_acc,
                'history':              history,
            }, periodic_ckpt, tag=f'periodic  epoch={epoch+1}')

        # ── Early stopping ─────────────────────────────────────────────────────
        if val_acc >= target_acc:
            print(f"\n*** Target accuracy {target_acc}% reached at epoch {epoch+1}! ***")
            break

    total_time      = time.time() - start_time
    epochs_completed = epoch + 1 - start_epoch   # epochs run this session

    # =============================================================================
    # SAVE RESULTS
    # =============================================================================

    # 1. Final metrics
    final_metrics = {
        'config':               config,
        'final_train_accuracy': float(history['train_acc'][-1]),
        'final_val_accuracy':   float(history['val_acc'][-1]),
        'best_val_accuracy':    float(max(history['val_acc'])),
        'best_val_epoch':       int(history['val_acc'].index(max(history['val_acc'])) + 1),
        'final_train_loss':     float(history['train_loss'][-1]),
        'final_val_loss':       float(history['val_loss'][-1]),
        'total_params':         n_params,
        'training_time_seconds': round(total_time, 2),
        'training_time_str':    str(datetime.timedelta(seconds=int(total_time))),
        'avg_epoch_time':       round(np.mean(history['epoch_times']), 2),
    }

    with open(os.path.join(config['output_dir'], 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nFinal Metrics saved to {config['output_dir']}/metrics.json")
    print(f"Best Val Accuracy: {final_metrics['best_val_accuracy']:.2f}% (Epoch {final_metrics['best_val_epoch']})")
    print(f"Total Training Time: {final_metrics['training_time_str']}")

    # 2. Training time summary
    time_summary = {
        'total_time_seconds':     total_time,
        'total_time_str':         str(datetime.timedelta(seconds=int(total_time))),
        'avg_epoch_time_seconds': float(np.mean(history['epoch_times'])),
        'epochs_completed':       epochs_completed,
        'epoch_times':            history['epoch_times'],
    }
    with open(os.path.join(config['output_dir'], 'training_time.json'), 'w') as f:
        json.dump(time_summary, f, indent=2)

    # 3. Loss / Accuracy curves
    total_epochs = len(history['train_loss'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(range(1, total_epochs+1), history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(range(1, total_epochs+1), history['val_loss'],   'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(range(1, total_epochs+1), history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(range(1, total_epochs+1), history['val_acc'],   'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Training curves saved to {config['output_dir']}/training_curves.png")

    # 4. Validation sample visualisation (loaded from best model)
    print("\nLoading best model weights for validation samples...")
    best_state = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best_state['model_state_dict'])

    sample_stats = save_validation_samples(model, val_loader, config, device, num_samples=16)
    print(f"Sample accuracy: {sample_stats['accuracy']:.1f}% ({sample_stats['correct']}/{sample_stats['total']})")

    # 5. Full history JSON
    with open(os.path.join(config['output_dir'], 'history.json'), 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

    print("\nAll results saved to:", config['output_dir'])
    print(f"Best model weights : {best_ckpt}")

if __name__ == '__main__':
    main()