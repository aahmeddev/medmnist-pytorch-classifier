"""Training CLI for MedResNet on 2D MedMNIST benchmarks with ROC-AUC checkpointing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn

from src.data import get_dataloaders
from src.model import MedResNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, n_classes: int) -> dict:
    model.eval()
    all_targets = []
    all_probs = []

    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.numpy().reshape(-1))

    y_true = np.concatenate(all_targets).astype(int)
    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if n_classes == 2:
        auc = float(roc_auc_score(y_true, y_prob[:, 1]))
    else:
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))

    return {"accuracy": acc, "macro_f1": macro_f1, "roc_auc": auc}


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader, val_loader, test_loader, meta = get_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = MedResNet(
        in_channels=meta["n_channels"],
        num_classes=meta["n_classes"],
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=meta["class_weights"].to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{meta['dataset']}_best.pt"

    history = []
    best_val_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.squeeze(-1).long().to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * images.size(0)
            total_samples += images.size(0)

        scheduler.step()
        train_loss = running_loss / max(total_samples, 1)
        val_metrics = evaluate_loader(model, val_loader, device, meta["n_classes"])
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), **{k: round(v, 4) for k, v in val_metrics.items()}})

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_loss:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['macro_f1']:.4f} | val_auc={val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dataset": meta["dataset"],
                    "n_channels": meta["n_channels"],
                    "n_classes": meta["n_classes"],
                    "label_map": meta["label_map"],
                    "dropout": args.dropout,
                    "best_val_auc": best_val_auc,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_loader(model, test_loader, device, meta["n_classes"])
    print(
        f"Best checkpoint test metrics -> "
        f"acc={test_metrics['accuracy']:.4f} | macro_f1={test_metrics['macro_f1']:.4f} | roc_auc={test_metrics['roc_auc']:.4f}"
    )

    summary = {
        "dataset": meta["dataset"],
        "device": str(device),
        "epochs": args.epochs,
        "splits": meta["splits"],
        "best_val_auc": round(best_val_auc, 4),
        "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
        "history": history,
    }
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MedResNet on a MedMNIST 2D medical imaging dataset.")
    parser.add_argument("--dataset", type=str, default="pneumoniamnist", help="MedMNIST 2D dataset key (default: pneumoniamnist)")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate before classification head")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for MedMNIST data cache")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for model checkpoints")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for metrics and plots")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
