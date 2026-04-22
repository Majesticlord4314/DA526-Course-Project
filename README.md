# Swin Transformer CIFAR-10 Image Classification

This project implements a Swin Transformer model trained on the CIFAR-10 dataset using PyTorch. The Swin Transformer leverages a hierarchical structure with shifted windows for efficient visual representation learning, making it highly effective for image classification tasks.

## Architecture Overview

**Swin Transformer (Shifted Window Transformer)** uses:
- Hierarchical feature maps with progressive merging of patches
- Shifted window approach for cross-window communication
- Local self-attention within non-overlapping windows
- Outstanding performance on image classification benchmarks

## Software Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher
- **torchvision**: For dataset loading and transforms
- **Additional packages**: tqdm, matplotlib, numpy, Pillow

### Installation

```bash
pip install torch torchvision tqdm matplotlib numpy Pillow
```

## Project Structure

| File | Description |
|------|-------------|
| `train_swin_cifar10.py` | Main training script that initializes the Swin Transformer model, loads CIFAR-10 data, trains for 10 epochs, and saves the best model checkpoint along with training metrics and visualizations |
| `validate_swin_cifar10.py` | Evaluation script that loads a trained model, generates predictions on the test set, produces a confusion matrix, plots validation samples with predictions, and exports validation metrics |
| `Swin-Transformer-main/` | Microsoft Swin Transformer library containing model definitions (`models/swin_transformer.py`), utility functions (`utils.py`), configuration files, and official implementation |
| `outputs/swin_cifar10_demo/` | Generated directory containing `best_model.pth` (trained weights), `training_curves.png` (loss/accuracy plots), `confusion_matrix.png`, `validation_samples.png`, `history.json`, and `metrics.json` |

## How to Run

### 1. Training

To train the model from scratch:

```bash
python train_swin_cifar10.py
```

This will:
- Initialize Swin-Tiny model with CIFAR-10 specific modifications
- Train for 10 epochs with early stopping
- Save the best model checkpoint
- Generate training curves and history logs

### 2. Validation / Testing

To evaluate a trained model and visualize results:

```bash
python validate_swin_cifar10.py
```

This will:
- Load the best model from `outputs/swin_cifar10_demo/`
- Run inference on test set
- Generate confusion matrix visualization
- Plot sample predictions with actual vs predicted labels
- Export detailed metrics to JSON

## Dataset

The project uses **CIFAR-10**, which contains 60,000 32x32 color images across 10 classes:
- Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

Split: 50,000 training images / 10,000 test images

## Output Examples

After running training and validation, check `outputs/swin_cifar10_demo/` for:
- **Training curves** - Visual representation of loss and accuracy over epochs
- **Confusion matrix** - Class-wise prediction accuracy breakdown
- **Validation samples** - Grid of test images with predicted vs actual labels
- **Metrics JSON** - Precision, recall, F1-score per class