"""Evaluation script generating ROC curves, confusion matrix, and Grad-CAM overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve
import torch

from src.data import get_dataloaders
from src.gradcam import GradCAM
from src.model import MedResNet


def run_evaluation(checkpoint_path: str | Path, data_dir: str | Path = "data", results_dir: str | Path = "results") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    dataset = ckpt["dataset"]
    label_map = ckpt["label_map"]
    n_classes = int(ckpt["n_classes"])
    class_names = [label_map[i] for i in range(n_classes)]

    _, _, test_loader, _ = get_dataloaders(dataset_name=dataset, data_dir=data_dir, batch_size=64)

    model = MedResNet(
        in_channels=int(ckpt["n_channels"]),
        num_classes=n_classes,
        dropout=float(ckpt.get("dropout", 0.25)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_targets, all_probs = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy().reshape(-1))

    y_true = np.concatenate(all_targets).astype(int)
    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.argmax(y_prob, axis=1)

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{dataset.upper()} Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # 2. ROC Curve
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc_val = roc_auc_score(y_true, y_prob[:, 1])
        ax.plot(fpr, tpr, color="#1f77b4", lw=2.2, label=f"{class_names[1]} (AUC = {auc_val:.3f})")
    else:
        for idx, name in enumerate(class_names):
            binary_true = (y_true == idx).astype(int)
            fpr, tpr, _ = roc_curve(binary_true, y_prob[:, idx])
            auc_val = roc_auc_score(binary_true, y_prob[:, idx])
            ax.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1.2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{dataset.upper()} Held-Out Test ROC Curve")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    # 3. Grad-CAM Visualisation Grid (8 test samples)
    gradcam = GradCAM(model, model.target_conv_layer)
    sample_images, sample_targets = next(iter(test_loader))
    n_show = min(8, sample_images.size(0))

    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    for i in range(n_show):
        img = sample_images[i : i + 1].to(device)
        true_idx = int(sample_targets[i].item())
        heatmap, pred_idx, conf = gradcam.generate(img)

        vis = sample_images[i].numpy() * 0.5 + 0.5
        vis = np.clip(vis, 0.0, 1.0)
        ax = axes[i // 4, i % 4]
        if vis.shape[0] == 1:
            ax.imshow(vis[0], cmap="gray")
        else:
            ax.imshow(np.transpose(vis, (1, 2, 0)))
        ax.imshow(heatmap.numpy(), cmap="jet", alpha=0.42)
        ax.set_title(f"True: {class_names[true_idx]}\nPred: {class_names[pred_idx]} ({conf:.2f})", fontsize=8.5)
        ax.axis("off")

    gradcam.remove_hooks()
    fig.suptitle(f"{dataset.upper()} Grad-CAM Saliency Overlays (Held-Out Test Set)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "gradcam_grid.png", dpi=160)
    plt.close(fig)
    print(f"Saved evaluation plots to {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained MedResNet checkpoint and generate plots.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pneumoniamnist_best.pt", help="Path to checkpoint")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for MedMNIST data cache")
    parser.add_argument("--results-dir", type=str, default="results", help="Output directory for figures")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.checkpoint, args.data_dir, args.results_dir)
