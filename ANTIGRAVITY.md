# Running the model training in Google Antigravity

This guide replaces the Colab notebook. Everything now runs as Python scripts inside Antigravity, on your local machine.

## 1. Open the project in Antigravity

- Open Antigravity → **Open Folder** → pick this repo.
- The agent will see `requirements.txt`, `src/`, `models/best_model.pth`, and this file.

## 2. Create a virtual environment (once)

In the Antigravity terminal:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

If you have an NVIDIA GPU and want CUDA acceleration, install a CUDA build of torch instead of the plain one — pick the command for your CUDA version at https://pytorch.org/get-started/locally/.

## 3. Get the dataset onto your disk

The scripts expect this layout (same as in the Colab notebook):

```
dataset/
  DATASET/
    TRAIN/
      O/   (organic .jpg files)
      R/   (recyclable .jpg files)
    TEST/
      O/
      R/
```

You have two options:

**Option A — Kaggle CLI (recommended)**

```bash
pip install kaggle
# Put your kaggle.json in ~/.kaggle/ (Kaggle → Account → Create New API Token)
mkdir -p dataset && cd dataset
kaggle datasets download -d techsash/waste-classification-data
unzip waste-classification-data.zip
# The zip already contains a DATASET/ folder — you should now have dataset/DATASET/TRAIN/{O,R} etc.
cd ..
```

**Option B — Manual**

Download from https://www.kaggle.com/datasets/techsash/waste-classification-data, unzip it, and move so the layout matches the tree above.

If your dataset lives somewhere else, set `DATASET_DIR`:

```bash
export DATASET_DIR=/absolute/path/to/DATASET   # macOS/Linux
setx DATASET_DIR "C:\path\to\DATASET"          # Windows (new shell picks it up)
```

## 4. Train MobileNetV2 and ResNet18

```bash
python -m src.train              # trains both, saves weights + curves in ./outputs
python -m src.train --model mobilenet   # only MobileNetV2
python -m src.train --model resnet      # only ResNet18
```

**CPU-friendly quick run** (proves the pipeline end-to-end in a few minutes, uses smaller images + subset):

```bash
QUICK=1 python -m src.train
```

Auto-detected behaviour:
- GPU available → 224×224 images, full train set, 5 head epochs + 3 fine-tune epochs.
- CPU only or `QUICK=1` → 160×160 images, 2000 train / 800 test images, 2+1 epochs.

## 5. Evaluate everything and generate report charts

Requires `models/best_model.pth` (already in the repo) plus the two `.pth` files from step 4.

```bash
python -m src.evaluate
```

Writes into `./outputs/`:

- `08_model_comparison.png` — Baseline CNN vs MobileNetV2 vs ResNet18 (accuracy + F1)
- `09_roc_and_per_class.png` — ROC curve + per-class precision/recall/F1
- `10_gradcam.png` — 8 Grad-CAM heatmaps on random test images
- `11_training_curves_transfer.png` — training curves
- `comparison.json` — the same table as JSON, easy to paste into the report
- `baseline_cnn.pth`, `mobilenet_waste.pth`, `resnet_waste.pth` — saved weights

## 6. Talking to the Antigravity agent

If you'd rather ask the agent to do it:

> "Set up a venv, install requirements, then run `python -m src.train` and `python -m src.evaluate`. If there is no GPU, set QUICK=1."

The agent can watch the training output and re-run failing steps for you.

## Troubleshooting

- **`FileNotFoundError: ... /TRAIN`** → Dataset isn't at the default path. Set `DATASET_DIR` (see step 3).
- **`CUDA out of memory`** → Set `QUICK=1`, or edit `src/common.py` and drop `batch` to 8.
- **Very slow on CPU** → Use `QUICK=1`. Full-size training on CPU takes many hours; use it only to sanity-check the pipeline.
- **Kaggle download 403** → Your `~/.kaggle/kaggle.json` isn't set up; regenerate the API token in Kaggle → Account.
