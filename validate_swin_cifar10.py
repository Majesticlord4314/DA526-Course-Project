"""
Validation report script for Swin CIFAR-10
- Loads saved checkpoints
- Generates metrics summary for final report
- Displays sample validation predictions with images
"""

import os
import sys
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Add Swin-Transformer-main to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Swin-Transformer-main'))

from models.swin_transformer import SwinTransformer
from timm.loss import LabelSmoothingCrossEntropy

# =============================================================================
# CONFIG - must match training config
# =============================================================================
CONFIG = {
    'img_size': 32,
    'patch_size': 4,
    'in_chans': 3,
    'num_classes': 10,
    'embed_dim': 48,
    'depths': [2, 2, 2],
    'num_heads': [4, 8, 16],
    'window_size': 4,
    'mlp_ratio': 4,
    'qkv_bias': True,
    'drop_rate': 0.0,
    'drop_path_rate': 0.15,
    'batch_size': 128,
    'label_smoothing': 0.1,
    'train_subset_pct': 0.5,
    'val_subset_pct': 1.0,
    'seed': 42,
    'output_dir': os.path.join(os.path.dirname(__file__), 'outputs', 'swin_cifar10_demo'),
}

# =============================================================================
# MODEL
# =============================================================================
def create_model(config, device):
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
# DATA
# =============================================================================
def get_val_loader(config):
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'cifar10')
    val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_val)
    val_subset_size = int(len(val_dataset) * config['val_subset_pct'])
    val_subset = Subset(val_dataset, list(range(val_subset_size)))
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    return val_loader, len(val_subset)

# =============================================================================
# VALIDATION
# =============================================================================
@torch.no_grad()
def validate_full(model, loader, criterion, device):
    model.eval()
    losses, correct, total = [], 0, 0
    all_preds, all_targets, all_probs = [], [], []

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        losses.append(loss.item())
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (np.mean(losses), 100. * correct / total,
            np.array(all_preds), np.array(all_targets), np.array(all_probs))

# =============================================================================
# PER-CLASS METRICS
# =============================================================================
def compute_per_class_metrics(preds, targets, num_classes=10):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    results = []
    for c in range(num_classes):
        mask = (targets == c)
        total_c = mask.sum()
        correct_c = ((preds == c) & mask).sum()
        acc = 100. * correct_c / total_c if total_c > 0 else 0.0
        results.append({'class': class_names[c], 'total': int(total_c),
                        'correct': int(correct_c), 'accuracy': round(acc, 2)})
    return results

# =============================================================================
# CONFUSION MATRIX PLOT
# =============================================================================
def plot_confusion_matrix(preds, targets, output_dir, num_classes=10):
    from sklearn.metrics import confusion_matrix
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    cm = confusion_matrix(targets, preds, labels=range(num_classes))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap='Blues')
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Normalized Confusion Matrix')

    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(j, i, f'{cm_norm[i, j]:.2f}',
                          ha='center', va='center', color='white' if cm_norm[i, j] > 0.5 else 'black')

    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {path}")
    return path

# =============================================================================
# SAMPLE PREDICTIONS VISUALIZATION
# =============================================================================
def visualize_samples(model, val_loader, config, device, num_samples=32):
    model.eval()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    samples = []
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            for i in range(images.size(0)):
                samples.append({
                    'image': images[i].cpu(),
                    'target': targets[i].item(),
                    'predicted': predicted[i].item(),
                    'confidence': probs[i].max().item(),
                    'correct': predicted[i].item() == targets[i].item()
                })
            if len(samples) >= num_samples:
                break

    samples = samples[:num_samples]
    rows, cols = 4, 8
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    fig.suptitle('Validation Sample Predictions (Top: Correct, Bottom: Incorrect)', fontsize=14, fontweight='bold')

    correct_samples = [s for s in samples if s['correct']][:16]
    incorrect_samples = [s for s in samples if not s['correct']][:16]

    for idx, ax in enumerate(axes.flat):
        ax.axis('off')

        if idx < len(correct_samples):
            s = correct_samples[idx]
            img = torch.clamp(s['image'] * std + mean, 0, 1)
            ax.imshow(img.permute(1, 2, 0))
            ax.set_title(f"T:{class_names[s['target']]}\nP:{class_names[s['predicted']]}\n{s['confidence']:.2f}",
                        color='green', fontsize=7)
        elif idx - len(correct_samples) < len(incorrect_samples):
            s = incorrect_samples[idx - len(correct_samples)]
            img = torch.clamp(s['image'] * std + mean, 0, 1)
            ax.imshow(img.permute(1, 2, 0))
            ax.set_title(f"T:{class_names[s['target']]}\nP:{class_names[s['predicted']]}\n{s['confidence']:.2f}",
                        color='red', fontsize=7)

    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'validation_samples_report.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample predictions saved to {path}")
    return path

# =============================================================================
# MAIN REPORT
# =============================================================================
def main():
    config = CONFIG
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Check for checkpoints
    best_ckpt = os.path.join(config['output_dir'], 'best_model.pth')
    periodic_ckpt = os.path.join(config['output_dir'], 'checkpoint_latest.pth')

    if not os.path.exists(best_ckpt) and not os.path.exists(periodic_ckpt):
        print(f"ERROR: No checkpoint found in {config['output_dir']}")
        print("Please run train_swin_cifar10.py first to train and save a model.")
        return

    # Load checkpoint
    ckpt_path = best_ckpt if os.path.exists(best_ckpt) else periodic_ckpt
    print(f"\nLoading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Create and load model
    model = create_model(config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Get data
    val_loader, n_val = get_val_loader(config)
    criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing'])

    # Run validation
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    val_loss, val_acc, preds, targets, probs = validate_full(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # Per-class metrics
    print("\n" + "-"*60)
    print("PER-CLASS ACCURACY")
    print("-"*60)
    per_class = compute_per_class_metrics(preds, targets)
    print(f"{'Class':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-"*40)
    for r in per_class:
        print(f"{r['class']:<12} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>10.2f}%")

    # Top-k accuracy
    print("\n" + "-"*60)
    print("TOP-K ACCURACY")
    print("-"*60)
    for k in [1, 3, 5]:
        topk_acc = (probs.argsort(axis=1)[:, -k:] == targets[:, np.newaxis]).any(axis=1).mean() * 100
        print(f"Top-{k} Accuracy: {topk_acc:.2f}%")

    # Loss and confidence stats
    print("\n" + "-"*60)
    print("PREDICTION STATISTICS")
    print("-"*60)
    confidences = probs.max(axis=1)
    print(f"Mean Confidence: {confidences.mean():.4f}")
    print(f"Min Confidence:  {confidences.min():.4f}")
    print(f"Max Confidence: {confidences.max():.4f}")

    # Correct vs Incorrect confidence distribution
    correct_conf = confidences[preds == targets]
    incorrect_conf = confidences[preds != targets]
    print(f"\nCorrect predictions - Mean Confidence: {correct_conf.mean():.4f}")
    print(f"Incorrect predictions - Mean Confidence: {incorrect_conf.mean():.4f}")

    # Checkpoint info
    print("\n" + "-"*60)
    print("CHECKPOINT INFO")
    print("-"*60)
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best Val Accuracy: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")

    # Generate plots
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    plot_confusion_matrix(preds, targets, config['output_dir'])
    visualize_samples(model, val_loader, config, device, num_samples=32)

    # Save metrics JSON
    metrics = {
        'val_loss': float(val_loss),
        'val_accuracy': float(val_acc),
        'total_params': n_params,
        'num_val_samples': n_val,
        'per_class_accuracy': {r['class']: r['accuracy'] for r in per_class},
        'top_k_accuracy': {
            f'top_{k}': float((probs.argsort(axis=1)[:, -k:] == targets[:, np.newaxis]).any(axis=1).mean() * 100)
            for k in [1, 3, 5]
        },
        'mean_confidence': float(confidences.mean()),
        'correct_confidence': float(correct_conf.mean()),
        'incorrect_confidence': float(incorrect_conf.mean()),
        'checkpoint_path': ckpt_path,
        'best_val_acc_in_checkpoint': float(checkpoint.get('best_val_acc', 0)),
        'epoch_in_checkpoint': checkpoint.get('epoch', 'N/A'),
    }
    report_path = os.path.join(config['output_dir'], 'validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {report_path}")

    print("\n" + "="*60)
    print("REPORT COMPLETE")
    print("="*60)
    print(f"Output directory: {config['output_dir']}")
    print("Generated files:")
    print("  - confusion_matrix.png")
    print("  - validation_samples_report.png")
    print("  - validation_report.json")

if __name__ == '__main__':
    main()
