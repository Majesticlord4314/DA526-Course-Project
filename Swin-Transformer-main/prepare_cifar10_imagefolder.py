import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from torchvision.datasets import CIFAR10


def export_split(dataset, out_root, split_name, per_class):
    counts = defaultdict(int)
    split_dir = out_root / split_name
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    for image, label in dataset:
        class_name = dataset.classes[label]
        if counts[class_name] >= per_class:
            continue
        class_dir = split_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        image.save(class_dir / f"{counts[class_name]:04d}.png")
        counts[class_name] += 1
        if len(counts) == len(dataset.classes) and all(v >= per_class for v in counts.values()):
            break

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/cifar10_raw")
    parser.add_argument("--out-root", default="data/cifar10_imagefolder")
    parser.add_argument("--train-per-class", type=int, default=80)
    parser.add_argument("--val-per-class", type=int, default=20)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    train = CIFAR10(root=raw_root, train=True, download=True)
    val = CIFAR10(root=raw_root, train=False, download=True)

    train_counts = export_split(train, out_root, "train", args.train_per_class)
    val_counts = export_split(val, out_root, "val", args.val_per_class)

    print(f"Exported CIFAR-10 ImageFolder dataset to {out_root.resolve()}")
    print(f"Train images: {sum(train_counts.values())} across {len(train_counts)} classes")
    print(f"Val images: {sum(val_counts.values())} across {len(val_counts)} classes")


if __name__ == "__main__":
    main()
