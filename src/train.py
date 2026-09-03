"""Train MobileNetV2 and/or ResNet18 with transfer learning.

Usage:
    python -m src.train                  # both models, saves weights + curves
    python -m src.train --model mobilenet
    python -m src.train --model resnet
    QUICK=1 python -m src.train          # CPU-friendly small run
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch

from .common import (
    build_mobilenet, build_resnet18, count_params, fit, load_config,
    make_loaders, print_banner,
)


def _phase1_freeze(model: torch.nn.Module, kind: str) -> None:
    if kind == "mobilenet":
        for p in model.features.parameters():
            p.requires_grad = False
    else:  # resnet
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True


def _phase2_unfreeze(model: torch.nn.Module, kind: str) -> None:
    if kind == "mobilenet":
        for name, p in model.named_parameters():
            if any(k in name for k in ("features.14", "features.15", "features.16",
                                        "features.17", "features.18", "classifier")):
                p.requires_grad = True
    else:  # resnet
        for name, p in model.named_parameters():
            if name.startswith("layer4") or name.startswith("fc"):
                p.requires_grad = True


def train_one(kind: str, cfg, train_loader, test_loader, class_weights=None) -> tuple[torch.nn.Module, dict]:
    build = build_mobilenet if kind == "mobilenet" else build_resnet18
    model = build(num_classes=cfg.num_classes).to(cfg.device)

    _phase1_freeze(model, kind)
    print(f"\n[{kind}] Phase 1 — head only (epochs={cfg.epochs_head}, lr=1e-3)")
    h1, _ = fit(model, train_loader, test_loader, cfg.epochs_head, 1e-3, cfg.device, tag=f"{kind}/head", class_weights=class_weights)

    _phase2_unfreeze(model, kind)
    print(f"\n[{kind}] Phase 2 — fine-tune (epochs={cfg.epochs_finetune}, lr=1e-4)")
    h2, _ = fit(model, train_loader, test_loader, cfg.epochs_finetune, 1e-4, cfg.device, tag=f"{kind}/ft", class_weights=class_weights)

    history = {k: h1[k] + h2[k] for k in h1}
    return model, history


def plot_curves(histories: dict[str, dict], out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, h in histories.items():
        axes[0].plot(h["train_loss"], label=f"{name} train")
        axes[0].plot(h["val_loss"], label=f"{name} val", linestyle="--")
        axes[1].plot(h["train_acc"], label=f"{name} train")
        axes[1].plot(h["val_acc"], label=f"{name} val", linestyle="--")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mobilenet", "resnet", "both"], default="both")
    args = ap.parse_args()

    cfg = load_config()
    os.makedirs(cfg.out_dir, exist_ok=True)
    print_banner(cfg)

    train_loader, test_loader, class_weights = make_loaders(cfg, img_size=cfg.img_size_transfer)
    print(f"Batches: train={len(train_loader)}  test={len(test_loader)}")

    kinds = ["mobilenet", "resnet"] if args.model == "both" else [args.model]
    histories: dict[str, dict] = {}

    for kind in kinds:
        model, hist = train_one(kind, cfg, train_loader, test_loader, class_weights=class_weights)
        histories[kind] = hist
        weights_path = os.path.join(cfg.out_dir, f"{kind}_waste.pth")
        torch.save(model.state_dict(), weights_path)
        print(f"[{kind}] saved -> {weights_path}  ({count_params(model):,} params)")

    hist_path = os.path.join(cfg.out_dir, "training_histories.json")
    with open(hist_path, "w") as f:
        json.dump(histories, f, indent=2)
    plot_curves(histories, os.path.join(cfg.out_dir, "11_training_curves_transfer.png"))
    print("Done. Wrote:", cfg.out_dir)


if __name__ == "__main__":
    main()
