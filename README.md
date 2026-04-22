# Swin Transformer CIFAR-10 Training

A PyTorch implementation of Swin Transformer for CIFAR-10 image classification.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- tqdm
- matplotlib
- numpy

Install dependencies:
```bash
pip install torch torchvision tqdm matplotlib numpy
```

## How to Run

### Training
```bash
python train_swin_cifar10.py
```

### Validation
```bash
python validate_swin_cifar10.py
```

## Files

- `train_swin_cifar10.py` - Training script with model saving and metrics
- `validate_swin_cifar10.py` - Validation with confusion matrix and visualizations
- `Swin-Transformer-main/` - Official Swin Transformer implementation
- `outputs/swin_cifar10_demo/` - Training results (curves, metrics, reports)