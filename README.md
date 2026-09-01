# AI-Waste-Classification

PBL project — AI-based waste classification using deep learning.

## Contents

- `notebooks/PBL_Waste_Classification_Continuation.ipynb` — **Colab notebook to continue model training** from the baseline. Trains MobileNetV2 + ResNet18 with transfer learning, produces a comparison table, ROC curve, per-class metrics and Grad-CAM heatmaps.
- `models/best_model.pth` — trained baseline (5-layer CNN, 128×128 input, ~14 MB).
- `PROJECT_CONTEXT.md` — full project background (datasets, research gaps, prior tasks).
- `outputs/` — populated by the continuation notebook (comparison PNGs, model checkpoints).

## How to run the continuation on Colab

1. Open Colab → **Runtime → Change runtime type → T4 GPU**.
2. Upload `notebooks/PBL_Waste_Classification_Continuation.ipynb`.
3. Upload `models/best_model.pth` into `/content/` (file panel on the left).
4. Make sure your Kaggle waste dataset is at `/content/dataset/dataset/DATASET/{TRAIN,TEST}/{O,R}` (same layout as the original notebook).
5. Run cells top-to-bottom. Total time on T4 ≈ 15–25 min.
6. The last cell zips all outputs and auto-downloads `pbl_continuation_outputs.zip`.

## What the continuation adds (mapped to your research gaps)

| Addition | Research gap addressed |
|---|---|
| MobileNetV2 transfer learning | #3 — lightweight mobile-deployable model |
| ResNet18 transfer learning | #3 — comparison point |
| Grad-CAM heatmaps | #4 — explainability (XAI) |
| ROC curve + per-class metrics | polish for the report |

## Baseline (already trained)

- 5-layer CNN, 128×128 input, 10 epochs, Adam(lr=1e-3), StepLR
- ~3.6 M params, ~14 MB
- Full details in `PROJECT_CONTEXT.md`
