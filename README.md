# AI-Waste-Classification

PBL project — AI-based waste classification using deep learning (Organic vs Recyclable).

## Primary workflow: **Antigravity** (local)

See [`ANTIGRAVITY.md`](ANTIGRAVITY.md) for the full setup guide. Short version:

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Put the Kaggle waste-classification dataset at ./dataset/DATASET/{TRAIN,TEST}/{O,R}
# Then:
python -m src.train        # trains MobileNetV2 + ResNet18 with transfer learning
python -m src.evaluate     # comparison + ROC + Grad-CAM + per-class metrics

# No GPU?  QUICK=1 python -m src.train
```

Outputs land in `./outputs/`.

## Contents

- `src/common.py` — config, dataloaders, model builders, train/eval helpers
- `src/train.py` — trains MobileNetV2 and/or ResNet18 (two-phase: freeze then fine-tune)
- `src/evaluate.py` — baseline eval, comparison table + chart, ROC, per-class metrics, Grad-CAM
- `models/best_model.pth` — pre-trained baseline (5-layer CNN, 128×128, ~14 MB)
- `notebooks/PBL_Waste_Classification_Continuation.ipynb` — same pipeline as a notebook (Colab or Antigravity's notebook view)
- `PROJECT_CONTEXT.md` — datasets, research gaps, prior tasks
- `ANTIGRAVITY.md` — step-by-step for Antigravity
- `requirements.txt` — Python dependencies

## What this continuation adds (mapped to research gaps)

| Addition | Research gap addressed |
|---|---|
| MobileNetV2 transfer learning | #3 — lightweight mobile-deployable model |
| ResNet18 transfer learning | #3 — comparison point |
| Grad-CAM heatmaps | #4 — explainability (XAI) |
| ROC curve + per-class metrics | polish for the report |

## Baseline (already trained, checked in)

5-layer CNN · 128×128 input · 10 epochs · Adam(lr=1e-3) · StepLR · ~3.6 M params · ~14 MB. Full details in `PROJECT_CONTEXT.md`.
