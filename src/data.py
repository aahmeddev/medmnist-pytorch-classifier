"""Reproducible MedMNIST 2D data loading and class-weight utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import medmnist
from medmnist import INFO
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def get_dataloaders(
    dataset_name: str = "pneumoniamnist",
    data_dir: str | Path = "data",
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Download (if needed) and build train/val/test DataLoaders for a MedMNIST 2D dataset."""
    key = dataset_name.lower()
    if key not in INFO:
        raise ValueError(f"Unknown MedMNIST dataset '{dataset_name}'. Available keys include pneumoniamnist, breastmnist, pathmnist, organamnist.")

    info = INFO[key]
    data_class = getattr(medmnist, info["python_class"])
    n_channels = int(info["n_channels"])
    n_classes = len(info["label"])

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    mean = [0.5] * n_channels
    std = [0.5] * n_channels

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_ds = data_class(split="train", transform=train_transform, download=True, root=str(root))
    val_ds = data_class(split="val", transform=eval_transform, download=True, root=str(root))
    test_ds = data_class(split="test", transform=eval_transform, download=True, root=str(root))

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    labels = np.asarray(train_ds.labels).reshape(-1).astype(int)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    class_weights = counts.sum() / (n_classes * counts)

    meta = {
        "dataset": key,
        "n_channels": n_channels,
        "n_classes": n_classes,
        "label_map": {int(k): v for k, v in info["label"].items()},
        "class_weights": torch.tensor(class_weights, dtype=torch.float32),
        "splits": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    }
    return train_loader, val_loader, test_loader, meta
