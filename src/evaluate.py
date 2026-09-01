"""Evaluate baseline + trained transfer models: comparison, ROC, per-class, Grad-CAM.

Prereqs (in cfg.out_dir): mobilenet_waste.pth, resnet_waste.pth  (created by src.train)
Baseline weights: cfg.baseline_pth

Usage:
    python -m src.evaluate
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (auc, classification_report,
                             precision_recall_fscore_support, roc_curve)

from .common import (
    CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, WasteClassifierCNN,
    build_mobilenet, build_resnet18, count_params, evaluate, load_config,
    make_loaders, print_banner, size_mb,
)


# ---------- Grad-CAM ----------

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        out = self.model(x)
        if class_idx is None:
            class_idx = int(out.argmax(1).item())
        self.model.zero_grad()
        out[0, class_idx].backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, torch.softmax(out, dim=1)[0, class_idx].item()


def denorm(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


# ---------- charts ----------

def plot_comparison(rows, out_path: str) -> None:
    labels = [r[0] for r in rows]
    accs = [r[3] for r in rows]
    f1s = [r[6] for r in rows]
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, accs, w, label="Accuracy")
    ax.bar(x + w / 2, f1s, w, label="F1 (weighted)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Score")
    ax.set_title("Model comparison — test set")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close(fig)


def plot_roc_and_perclass(best_name: str, best_res: dict, out_path: str) -> None:
    y_true = best_res["y_true"]; y_pred = best_res["y_pred"]
    fpr, tpr, _ = roc_curve(y_true, best_res["y_prob"][:, 1])
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(fpr, tpr, label=f"{best_name} (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — Recyclable class"); axes[0].legend(); axes[0].grid(alpha=0.3)

    prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    x = np.arange(2); w = 0.25
    axes[1].bar(x - w, prec_c, w, label="Precision")
    axes[1].bar(x, rec_c, w, label="Recall")
    axes[1].bar(x + w, f1_c, w, label="F1")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"{CLASS_NAMES[0]} (O)", f"{CLASS_NAMES[1]} (R)"])
    axes[1].set_ylim(0, 1.05); axes[1].set_title(f"Per-class metrics — {best_name}")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close(fig)


def plot_gradcam(model: nn.Module, target_layer: nn.Module, test_ds, device, out_path: str, n: int = 8) -> None:
    gradcam = GradCAM(model, target_layer)
    idxs = np.random.RandomState(42).choice(len(test_ds), size=n, replace=False)
    rows = (n + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    for ax, i in zip(axes.ravel(), idxs):
        x, y = test_ds[i]
        cam, pred_idx, conf = gradcam(x.unsqueeze(0).to(device))
        ax.imshow(denorm(x))
        ax.imshow(cam, cmap="jet", alpha=0.45)
        color = "green" if pred_idx == y else "red"
        ax.set_title(f"True: {CLASS_NAMES[y]}\nPred: {CLASS_NAMES[pred_idx]} ({conf:.2f})",
                     color=color, fontsize=10)
        ax.axis("off")
    plt.suptitle("Grad-CAM heatmaps — MobileNetV2", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close(fig)


# ---------- main ----------

def _underlying_dataset(loader):
    ds = loader.dataset
    from torch.utils.data import Subset
    if isinstance(ds, Subset):
        return ds.dataset  # base ImageFolder for indexed access; keep loader for eval
    return ds


def main() -> None:
    cfg = load_config()
    os.makedirs(cfg.out_dir, exist_ok=True)
    print_banner(cfg)

    # Baseline needs 128x128; transfer models need the trained size (224 or 160)
    _, test_loader_128 = make_loaders(cfg, img_size=128)
    _, test_loader_tx = make_loaders(cfg, img_size=cfg.img_size_transfer)

    # --- baseline ---
    if not os.path.exists(cfg.baseline_pth):
        raise FileNotFoundError(f"Baseline weights not found at {cfg.baseline_pth}")
    baseline = WasteClassifierCNN().to(cfg.device)
    state = torch.load(cfg.baseline_pth, map_location=cfg.device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    baseline.load_state_dict(state)
    baseline_res = evaluate(baseline, test_loader_128, cfg.device, nn.CrossEntropyLoss())
    print(f"Baseline CNN — acc={baseline_res['acc']:.4f}  f1={baseline_res['f1']:.4f}")

    # --- transfer models ---
    mob_pth = os.path.join(cfg.out_dir, "mobilenet_waste.pth")
    res_pth = os.path.join(cfg.out_dir, "resnet_waste.pth")
    if not os.path.exists(mob_pth) or not os.path.exists(res_pth):
        raise FileNotFoundError(f"Run `python -m src.train` first (need {mob_pth} and {res_pth}).")

    mobilenet = build_mobilenet().to(cfg.device)
    mobilenet.load_state_dict(torch.load(mob_pth, map_location=cfg.device))
    mob_res = evaluate(mobilenet, test_loader_tx, cfg.device, nn.CrossEntropyLoss())
    print(f"MobileNetV2 — acc={mob_res['acc']:.4f}  f1={mob_res['f1']:.4f}")

    resnet = build_resnet18().to(cfg.device)
    resnet.load_state_dict(torch.load(res_pth, map_location=cfg.device))
    res_res = evaluate(resnet, test_loader_tx, cfg.device, nn.CrossEntropyLoss())
    print(f"ResNet18    — acc={res_res['acc']:.4f}  f1={res_res['f1']:.4f}")

    # --- comparison table ---
    baseline_saved = os.path.join(cfg.out_dir, "baseline_cnn.pth")
    torch.save(baseline.state_dict(), baseline_saved)
    rows = [
        ("Baseline CNN (5-layer)", count_params(baseline), size_mb(baseline_saved),
         baseline_res["acc"], baseline_res["precision"], baseline_res["recall"], baseline_res["f1"]),
        ("MobileNetV2 (transfer)", count_params(mobilenet), size_mb(mob_pth),
         mob_res["acc"], mob_res["precision"], mob_res["recall"], mob_res["f1"]),
        ("ResNet18 (transfer)", count_params(resnet), size_mb(res_pth),
         res_res["acc"], res_res["precision"], res_res["recall"], res_res["f1"]),
    ]
    print()
    print(f"{'Model':<26}{'Params':>12}{'Size(MB)':>12}{'Acc':>10}{'Prec':>10}{'Rec':>10}{'F1':>10}")
    for r in rows:
        print(f"{r[0]:<26}{r[1]:>12,}{r[2]:>12.2f}{r[3]:>10.4f}{r[4]:>10.4f}{r[5]:>10.4f}{r[6]:>10.4f}")

    with open(os.path.join(cfg.out_dir, "comparison.json"), "w") as f:
        json.dump([{"model": r[0], "params": r[1], "size_mb": r[2],
                    "accuracy": r[3], "precision": r[4], "recall": r[5], "f1": r[6]} for r in rows], f, indent=2)
    plot_comparison(rows, os.path.join(cfg.out_dir, "08_model_comparison.png"))

    # --- ROC + per-class on best transfer model ---
    best_name, best_res = ("MobileNetV2", mob_res) if mob_res["acc"] >= res_res["acc"] else ("ResNet18", res_res)
    plot_roc_and_perclass(best_name, best_res, os.path.join(cfg.out_dir, "09_roc_and_per_class.png"))
    print(f"\nClassification report ({best_name}):")
    print(classification_report(best_res["y_true"], best_res["y_pred"], target_names=CLASS_NAMES, digits=4))

    # --- Grad-CAM on MobileNetV2 ---
    test_ds = _underlying_dataset(test_loader_tx)
    plot_gradcam(mobilenet, mobilenet.features[-1], test_ds, cfg.device,
                 os.path.join(cfg.out_dir, "10_gradcam.png"))

    print("Done. Charts + JSON in:", cfg.out_dir)


if __name__ == "__main__":
    main()
