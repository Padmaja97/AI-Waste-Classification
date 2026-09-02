"""Evaluate baseline + trained transfer models: comparison, ROC, per-class, Grad-CAM.

Prereqs (in cfg.out_dir): mobilenet_waste.pth, resnet_waste.pth  (created by src.train)
Baseline weights: cfg.baseline_pth

Usage:
    python -m src.evaluate
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_curve)

from .common import (
    CLASS_NAMES, DIR_TO_NAME, IMAGENET_MEAN, IMAGENET_STD, WasteClassifierCNN,
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


def plot_roc_and_perclass(best_name: str, best_res: dict, out_path: str, class_names: list[str] | None = None) -> None:
    if class_names is None:
        class_names = CLASS_NAMES
    n_cls = len(class_names)
    y_true = best_res["y_true"]; y_pred = best_res["y_pred"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if n_cls == 2:
        fpr, tpr, _ = roc_curve(y_true, best_res["y_prob"][:, 1])
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{best_name} (AUC = {roc_auc:.3f})")
        axes[0].set_title(f"ROC Curve — {class_names[1]} class")
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=list(range(n_cls)))
        for i in range(n_cls):
            fpr, tpr, _ = roc_curve(y_bin[:, i], best_res["y_prob"][:, i])
            roc_auc = auc(fpr, tpr)
            axes[0].plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")
        axes[0].set_title(f"ROC Curves (one-vs-rest)")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    labels = list(range(n_cls))
    prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    x = np.arange(n_cls); w = 0.25
    axes[1].bar(x - w, prec_c, w, label="Precision")
    axes[1].bar(x, rec_c, w, label="Recall")
    axes[1].bar(x + w, f1_c, w, label="F1")
    dir_keys = sorted(k for k, v in DIR_TO_NAME.items() if v in class_names)
    tick_labels = [f"{class_names[i]} ({dir_keys[i]})" for i in range(n_cls)]
    axes[1].set_xticks(x); axes[1].set_xticklabels(tick_labels, rotation=15 if n_cls > 2 else 0)
    axes[1].set_ylim(0, 1.05); axes[1].set_title(f"Per-class metrics — {best_name}")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close(fig)


def plot_gradcam(model: nn.Module, target_layer: nn.Module, test_ds, device, out_path: str, class_names: list[str] | None = None, n: int = 8) -> None:
    if class_names is None:
        class_names = CLASS_NAMES
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
        true_name = class_names[y] if y < len(class_names) else f"class_{y}"
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"
        ax.set_title(f"True: {true_name}\nPred: {pred_name} ({conf:.2f})",
                     color=color, fontsize=10)
        ax.axis("off")
    plt.suptitle("Grad-CAM heatmaps — MobileNetV2", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close(fig)


# ---------- measured latency ----------

@torch.no_grad()
def benchmark_latency(model: nn.Module, device, img_size: int, runs: int = 60, warmup: int = 12) -> dict:
    """Time single-image forward passes. Reported as median + p95, never a round claim."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    return {
        "median_ms": round(statistics.median(samples), 2),
        "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 2),
        "min_ms": round(samples[0], 2),
        "runs": runs,
        "batch_size": 1,
        "input_size": img_size,
        "device": str(device),
    }


def _per_class(y_true, y_pred, class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        class_names[i]: {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f[i]), 4),
            "support": int(s[i]),
        }
        for i in labels
    }


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
    nc = cfg.num_classes
    cnames = cfg.class_names

    baseline = WasteClassifierCNN(num_classes=nc).to(cfg.device)
    state = torch.load(cfg.baseline_pth, map_location=cfg.device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        baseline.load_state_dict(state)
    except RuntimeError:
        print(f"Baseline weights are for a different class count — retraining needed.")
        baseline = WasteClassifierCNN(num_classes=nc).to(cfg.device)
    baseline_res = evaluate(baseline, test_loader_128, cfg.device, nn.CrossEntropyLoss())
    print(f"Baseline CNN — acc={baseline_res['acc']:.4f}  f1={baseline_res['f1']:.4f}")

    # --- transfer models ---
    mob_pth = os.path.join(cfg.out_dir, "mobilenet_waste.pth")
    res_pth = os.path.join(cfg.out_dir, "resnet_waste.pth")
    if not os.path.exists(mob_pth) or not os.path.exists(res_pth):
        raise FileNotFoundError(f"Run `python -m src.train` first (need {mob_pth} and {res_pth}).")

    mobilenet = build_mobilenet(num_classes=nc).to(cfg.device)
    mobilenet.load_state_dict(torch.load(mob_pth, map_location=cfg.device))
    mob_res = evaluate(mobilenet, test_loader_tx, cfg.device, nn.CrossEntropyLoss())
    print(f"MobileNetV2 — acc={mob_res['acc']:.4f}  f1={mob_res['f1']:.4f}")

    resnet = build_resnet18(num_classes=nc).to(cfg.device)
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

    # ---- full provenance report: every number the web UI shows comes from here ----
    print("\nBenchmarking single-image latency...")
    entries = [
        ("Baseline CNN", "baseline", baseline, baseline_res, baseline_saved, 128),
        ("MobileNetV2", "mobilenet", mobilenet, mob_res, mob_pth, cfg.img_size_transfer),
        ("ResNet18", "resnet", resnet, res_res, res_pth, cfg.img_size_transfer),
    ]

    cls_labels = list(range(nc))

    models_out = []
    for name, key, mdl, res, pth, isize in entries:
        cmatrix = confusion_matrix(res["y_true"], res["y_pred"], labels=cls_labels)
        lat = benchmark_latency(mdl, cfg.device, isize)
        print(f"  {name:<16} median {lat['median_ms']:>7.2f} ms   p95 {lat['p95_ms']:>7.2f} ms")
        models_out.append({
            "name": name,
            "key": key,
            "input_size": isize,
            "params": count_params(mdl),
            "size_mb": round(size_mb(pth), 2),
            "accuracy": round(float(res["acc"]), 4),
            "precision": round(float(res["precision"]), 4),
            "recall": round(float(res["recall"]), 4),
            "f1": round(float(res["f1"]), 4),
            "per_class": _per_class(res["y_true"], res["y_pred"], cnames),
            "confusion": cmatrix.tolist(),
            "latency": lat,
        })

    def _count(split: str) -> dict:
        root = os.path.join(cfg.dataset_dir, split)
        out = {}
        for cls_dir in sorted(DIR_TO_NAME.keys()):
            cls_name = DIR_TO_NAME[cls_dir]
            if cls_name in cnames:
                p = os.path.join(root, cls_dir)
                out[cls_name] = len(os.listdir(p)) if os.path.isdir(p) else 0
        out["total"] = sum(v for k, v in out.items() if k != "total")
        return out

    report = {
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "classes": cnames,
        "num_classes": nc,
        "run": {
            "quick_mode": bool(cfg.quick),
            "device": str(cfg.device),
            "transfer_input_size": cfg.img_size_transfer,
            "baseline_input_size": 128,
            "epochs_head": cfg.epochs_head,
            "epochs_finetune": cfg.epochs_finetune,
            "batch_size": cfg.batch,
            "train_subset": cfg.subset_train,
            "test_subset": cfg.subset_test,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "machine": platform.platform(),
        },
        "dataset": {
            "root": cfg.dataset_dir,
            "train": _count("TRAIN"),
            "test": _count("TEST"),
        },
        "evaluation": {
            "test_images_used": int(len(mob_res["y_true"])),
            "note": (
                "QUICK mode: trained and evaluated on a random subset, not the full dataset."
                if cfg.quick else
                "Full run: trained on the complete training set and evaluated on the full test set."
            ),
        },
        "models": models_out,
        "best": max(models_out, key=lambda m: m["accuracy"])["name"],
    }
    report_path = os.path.join(cfg.out_dir, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote provenance report -> {report_path}")

    # --- ROC + per-class on best transfer model ---
    best_name, best_res = ("MobileNetV2", mob_res) if mob_res["acc"] >= res_res["acc"] else ("ResNet18", res_res)
    plot_roc_and_perclass(best_name, best_res, os.path.join(cfg.out_dir, "09_roc_and_per_class.png"), cnames)
    print(f"\nClassification report ({best_name}):")
    print(classification_report(best_res["y_true"], best_res["y_pred"], target_names=cnames, digits=4))

    # --- Grad-CAM on MobileNetV2 ---
    test_ds = _underlying_dataset(test_loader_tx)
    plot_gradcam(mobilenet, mobilenet.features[-1], test_ds, cfg.device,
                 os.path.join(cfg.out_dir, "10_gradcam.png"), cnames)

    print("Done. Charts + JSON in:", cfg.out_dir)


if __name__ == "__main__":
    main()
