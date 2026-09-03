"""EcoSort AI — Flask backend for 4-class waste classification.

Loads MobileNetV2 (preferred) or baseline CNN, serves predictions with
Grad-CAM, and reads all measured metrics from outputs/eval_report.json.

Usage:
    cd AI-Waste-Classification
    python -m webapp.app              # or: python webapp/app.py
"""
from __future__ import annotations

import base64
import io
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms

# ── paths (work whether run from project root or webapp/) ──────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs")
BASELINE_DIR = os.path.join(PROJECT_ROOT, "models")

# ── app ────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# ── metrics routes ─────────────────────────────────────────────────
import sys
sys.path.insert(0, BASE_DIR)
from metrics_routes import register_metrics
register_metrics(app, outputs_dir=OUTPUTS_DIR)

# ── class config ───────────────────────────────────────────────────
DIR_TO_NAME = {"E": "E-waste", "H": "Hazardous", "N": "Non-Recyclable", "O": "Organic", "R": "Recyclable"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}


def _detect_classes() -> list[str]:
    ds_dir = os.environ.get("DATASET_DIR", os.path.join(PROJECT_ROOT, "dataset", "DATASET"))
    train_dir = os.path.join(ds_dir, "TRAIN")
    if os.path.isdir(train_dir):
        dirs = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
        names = []
        for d in dirs:
            if d not in DIR_TO_NAME:
                continue
            d_path = os.path.join(train_dir, d)
            has_imgs = any(os.path.splitext(f)[1].lower() in IMAGE_EXTS
                          for f in os.listdir(d_path)
                          if os.path.isfile(os.path.join(d_path, f)))
            if has_imgs:
                names.append(DIR_TO_NAME[d])
        if len(names) >= 2:
            return names
    return ["Organic", "Recyclable"]

CLASSES = _detect_classes()
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── model loading ──────────────────────────────────────────────────
model = None
model_kind = None
gradcam_layer = None


def _build_mobilenet(nc: int):
    from torchvision import models
    m = models.mobilenet_v2(weights=None)
    m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, nc)
    return m


def _build_baseline(nc: int):
    from torch import nn
    class CNN(nn.Module):
        def __init__(self):
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
                nn.Linear(256, nc),
            )
        def forward(self, x):
            return self.classifier(self.features(x))
    return CNN()


def _build_resnet18(nc: int):
    from torchvision import models
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, nc)
    return m


def _load_model():
    global model, model_kind, gradcam_layer

    candidates = [
        (os.path.join(MODELS_DIR, "mobilenet_waste.pth"), "mobilenet"),
        (os.path.join(MODELS_DIR, "resnet_waste.pth"), "resnet"),
        (os.path.join(BASELINE_DIR, "best_model.pth"), "baseline"),
    ]

    for path, kind in candidates:
        if not os.path.isfile(path):
            continue
        try:
            state = torch.load(path, map_location=DEVICE, weights_only=True)
            nc = NUM_CLASSES

            if kind == "mobilenet":
                m = _build_mobilenet(nc)
                gradcam_layer = m.features[-1]
            elif kind == "resnet":
                m = _build_resnet18(nc)
                gradcam_layer = m.layer4
            else:
                m = _build_baseline(nc)
                gradcam_layer = m.features[-4]

            m.load_state_dict(state, strict=False)
            m.to(DEVICE).eval()
            model = m
            model_kind = kind
            print(f"Loaded {kind} from {path} ({nc} classes, {DEVICE})")
            return
        except Exception as e:
            print(f"Could not load {path}: {e}")

    print("No model weights found — /api/predict will return 503.")


_load_model()


def _get_transform():
    if model_kind == "baseline":
        size = 128
    else:
        size = 224
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


transform = _get_transform()

# ── helpers ────────────────────────────────────────────────────────

def _read_upload() -> Image.Image:
    if "image" in request.files:
        return Image.open(request.files["image"].stream).convert("RGB")
    b64 = request.form.get("image_b64", "")
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _gradcam_png(mdl, x, class_idx, pil_img):
    acts, grads = {}, {}
    layer = gradcam_layer
    if layer is None:
        print("Grad-CAM: no target layer set")
        return None

    h1 = layer.register_forward_hook(lambda m, i, o: acts.setdefault("v", o))
    try:
        h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.setdefault("v", go[0]))
    except AttributeError:
        h2 = layer.register_backward_hook(lambda m, gi, go: grads.setdefault("v", go[0]))

    was_training = mdl.training
    mdl.train()
    mdl.zero_grad()
    out = mdl(x)
    out[0, class_idx].backward()
    h1.remove()
    h2.remove()
    if not was_training:
        mdl.eval()

    if "v" not in grads or "v" not in acts:
        print(f"Grad-CAM: hooks did not fire (acts={'v' in acts}, grads={'v' in grads})")
        return None

    w = grads["v"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts["v"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(pil_img.height, pil_img.width),
                        mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    rgb = (np.dstack([r, g, b]) * 255).astype("uint8")

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── routes ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify(
        model_loaded=model is not None,
        model_kind=model_kind,
        device=str(DEVICE),
        classes=CLASSES,
        num_classes=NUM_CLASSES,
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(error="No model loaded. Train first: python -m src.train"), 503

    try:
        img = _read_upload()
    except Exception:
        return jsonify(error="Could not read the image."), 400

    x = transform(img).unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    elapsed = (time.perf_counter() - t0) * 1000

    idx = int(probs.argmax())

    try:
        x_grad = transform(img).unsqueeze(0).to(DEVICE).requires_grad_(True)
        cam = _gradcam_png(model, x_grad, idx, img)
    except Exception as e:
        import traceback
        print(f"Grad-CAM error: {e}")
        traceback.print_exc()
        cam = None

    prob_dict = {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))}

    return jsonify(
        label=CLASSES[idx],
        class_index=idx,
        confidence=round(float(probs[idx]), 4),
        probabilities=prob_dict,
        inference_ms=round(elapsed, 1),
        gradcam=cam,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
