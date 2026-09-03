"""Shared config, data loaders, training/eval helpers.

Environment variables you can set (defaults in parentheses):
    DATASET_DIR   root folder containing TRAIN/ and TEST/         (./dataset/DATASET)
    OUT_DIR       where to write charts, models, JSON             (./outputs)
    BASELINE_PTH  path to the pre-trained baseline CNN weights    (./models/best_model.pth)
    QUICK         "1" for tiny/CPU-friendly run                   (0)
"""
from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DIR_TO_NAME = {"E": "E-waste", "H": "Hazardous", "N": "Non-Recyclable", "O": "Organic", "R": "Recyclable"}
CLASS_NAMES = ["Organic", "Recyclable"]


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}


def _has_images(folder: str) -> bool:
    """Return True if folder contains at least one image file."""
    if not os.path.isdir(folder):
        return False
    return any(os.path.splitext(f)[1].lower() in IMAGE_EXTS
               for f in os.listdir(folder)
               if os.path.isfile(os.path.join(folder, f)))


def detect_classes(dataset_dir: str) -> list[str]:
    """Auto-detect class names from non-empty TRAIN subdirectories."""
    train_dir = os.path.join(dataset_dir, "TRAIN")
    if not os.path.isdir(train_dir):
        return CLASS_NAMES
    dirs = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    names = [DIR_TO_NAME[d] for d in dirs
             if d in DIR_TO_NAME and _has_images(os.path.join(train_dir, d))]
    return names if len(names) >= 2 else CLASS_NAMES


@dataclass
class Config:
    dataset_dir: str
    out_dir: str
    baseline_pth: str
    quick: bool
    device: torch.device
    batch: int
    img_size_transfer: int      # 224 for full, 160 for quick
    epochs_head: int
    epochs_finetune: int
    subset_train: int | None    # None = use all
    subset_test: int | None
    class_names: list[str] | None = None

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = detect_classes(self.dataset_dir)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def train_dir(self) -> str:
        return os.path.join(self.dataset_dir, "TRAIN")

    @property
    def test_dir(self) -> str:
        return os.path.join(self.dataset_dir, "TEST")


def load_config() -> Config:
    quick = os.environ.get("QUICK", "0") == "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if quick or device.type == "cpu":
        img_size, eh, ef, sub_tr, sub_te = 160, 2, 1, 2000, 800
    else:
        img_size, eh, ef, sub_tr, sub_te = 224, 5, 3, None, None
    ds_dir = os.environ.get("DATASET_DIR", "./dataset/DATASET")
    return Config(
        dataset_dir=ds_dir,
        out_dir=os.environ.get("OUT_DIR", "./outputs"),
        baseline_pth=os.environ.get("BASELINE_PTH", "./models/best_model.pth"),
        quick=quick or device.type == "cpu",
        device=device,
        batch=32 if device.type == "cuda" else 16,
        img_size_transfer=img_size,
        epochs_head=eh,
        epochs_finetune=ef,
        subset_train=sub_tr,
        subset_test=sub_te,
        class_names=detect_classes(ds_dir),
    )


def print_banner(cfg: Config) -> None:
    print("=" * 60)
    print(f"Device       : {cfg.device}" + (f" ({torch.cuda.get_device_name(0)})" if cfg.device.type == "cuda" else ""))
    print(f"Dataset root : {cfg.dataset_dir}")
    print(f"Output dir   : {cfg.out_dir}")
    print(f"Baseline pth : {cfg.baseline_pth}")
    print(f"Quick mode   : {cfg.quick}  (img={cfg.img_size_transfer}, epochs={cfg.epochs_head}+{cfg.epochs_finetune})")
    print(f"Classes ({cfg.num_classes})  : {', '.join(cfg.class_names)}")
    print("=" * 60)


# ---------- data ----------

def make_transforms(img_size: int):
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf_train, tf_eval


def _maybe_subset(ds, n: int | None, seed: int = 0):
    if n is None or n >= len(ds):
        return ds
    idx = np.random.RandomState(seed).choice(len(ds), size=n, replace=False)
    return Subset(ds, idx.tolist())


def make_loaders(cfg: Config, img_size: int):
    if not os.path.isdir(cfg.train_dir) or not os.path.isdir(cfg.test_dir):
        raise FileNotFoundError(
            f"Expected {cfg.train_dir} and {cfg.test_dir}. "
            "Set DATASET_DIR or place the dataset there. See ANTIGRAVITY.md."
        )
    tf_train, tf_eval = make_transforms(img_size)
    train_ds = datasets.ImageFolder(cfg.train_dir, transform=tf_train)
    test_ds = datasets.ImageFolder(cfg.test_dir, transform=tf_eval)
    expected_dirs = sorted(k for k, v in DIR_TO_NAME.items() if v in cfg.class_names)
    if train_ds.classes != expected_dirs:
        print(f"Info: dataset classes {train_ds.classes} (expected {expected_dirs})")

    targets = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(targets, minlength=len(train_ds.classes))
    weights = len(targets) / (len(train_ds.classes) * class_counts.clip(min=1).astype(float))
    class_weights = torch.FloatTensor(weights)
    print(f"Class weights: {', '.join(f'{train_ds.classes[i]}={w:.2f}' for i, w in enumerate(class_weights))}")

    train_ds = _maybe_subset(train_ds, cfg.subset_train)
    test_ds = _maybe_subset(test_ds, cfg.subset_test)
    num_workers = 2 if cfg.device.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True,
                              num_workers=num_workers, pin_memory=cfg.device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=cfg.batch, shuffle=False,
                             num_workers=num_workers, pin_memory=cfg.device.type == "cuda")
    return train_loader, test_loader, class_weights


# ---------- models ----------

class WasteClassifierCNN(nn.Module):
    """The baseline 5-layer CNN used in Task 4 (128x128 input)."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_mobilenet(num_classes: int = 2) -> nn.Module:
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def build_resnet18(num_classes: int = 2) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


# ---------- train / eval ----------

def _step(model, x, y, criterion, optimizer=None):
    out = model(x)
    loss = criterion(out, y)
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item(), out


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss, out = _step(model, x, y, criterion, optimizer)
        running_loss += loss * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    running_loss, total = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if criterion is not None:
            running_loss += criterion(out, y).item() * x.size(0)
        probs = torch.softmax(out, dim=1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())
        y_prob.extend(probs.cpu().numpy())
        total += x.size(0)
    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "loss": running_loss / total if criterion is not None else None,
        "acc": acc, "precision": prec, "recall": rec, "f1": f1,
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
    }


def fit(model, train_loader, test_loader, epochs, lr, device, weight_decay=1e-4, tag="model", class_weights=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_state = 0.0, None
    for e in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device)
        v = evaluate(model, test_loader, device, criterion)
        scheduler.step()
        history["train_loss"].append(tl); history["train_acc"].append(ta)
        history["val_loss"].append(v["loss"]); history["val_acc"].append(v["acc"])
        if v["acc"] > best_acc:
            best_acc = v["acc"]
            best_state = copy.deepcopy(model.state_dict())
        print(f"[{tag}] {e}/{epochs}  train_loss={tl:.4f} train_acc={ta:.4f}  "
              f"val_loss={v['loss']:.4f} val_acc={v['acc']:.4f}  ({time.time()-t0:.1f}s)")
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_acc


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)
