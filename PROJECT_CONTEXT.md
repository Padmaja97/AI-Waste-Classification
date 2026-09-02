# PBL PROJECT — FULL CONTEXT FOR CLAUDE CODE

## WHO IS THE USER
- Name: Simran
- Course: BTech CSE, 2nd year, Symbiosis University
- Location: Nagpur, India
- Skill level: Beginner in ML/AI, knows React/Firebase/web dev

---

## PROJECT OVERVIEW
- **Subject:** PBL (Project-Based Learning)
- **Topic:** AI-Based Waste Classification Using Deep Learning
- **Goal:** Study research papers, identify research gaps, implement a CNN model to classify waste images

---

## TASKS COMPLETED

### Task 1: Dataset Selection (DONE)
Selected 2 datasets from a list of 25+ research papers (uploaded as `final_1_to_25_pbl.xlsx`):

**Dataset 1 — Waste Classification (Organic vs Recyclable)**
- Paper Number: #30
- Author: Nonso Nnamoko (March 2022)
- Paper Title: CNN-based waste image classification
- Dataset: Sekar's Waste Classification Dataset
- Size: 25,077 images (Organic: 13,966 | Recyclable: 11,111)
- Classes: 2 (Organic, Recyclable)
- Source: Kaggle — https://www.kaggle.com/datasets/techsash/waste-classification-data
- Paper Link: https://www.mdpi.com/2412-3811/7/4/47
- Original Model: Custom 5-layer CNN, Best Accuracy: 80.88% at 80x45 resolution

**Dataset 2 — E-Waste Detection**
- Paper Number: #10
- Author: Shubhyansh Rai (February 2026)
- Paper Title: EW YOLO — Edge Computing IoT and YOLOv11 for E-Waste Detection
- Dataset: E-Waste Dataset
- Size: 19,613 annotated images
- Classes: 77 (aligned with UNU-KEY categories — smartphones, laptops, keyboards, etc.)
- Source: Roboflow — https://universe.roboflow.com/electronic-waste-detection/e-waste-dataset-r0ojc
- Paper Link: https://www.mdpi.com/2076-3417/16/4/2152
- Original Model: YOLOv11, Best mAP@0.50: 0.90074

### Task 2: Research Gap Identification (DONE)
5 research gaps identified from analyzing 25 papers:

1. **No Unified Multi-Category Classification System**
   - Paper #30 does only Organic/Recyclable (2 classes)
   - Paper #10 does only E-Waste (77 classes)
   - No paper combines both into a single unified system
   - Related Papers: #30, #10, #2

2. **Lack of Indian Waste Composition Testing**
   - Most datasets from Western countries (UK, Ireland, Japan)
   - Indian waste has different composition (~60% organic, mixed plastics, coconut shells, jute)
   - Only Paper #31 (IIT Kharagpur) and #32 (Ujjain) address Indian context but without AI
   - Related Papers: #31, #32, #21

3. **No Lightweight Mobile-Deployable Models**
   - Paper #10's YOLOv11 needs GPU (5.2 MB)
   - Paper #30's CNN is 1 MB but only 80% accurate
   - No paper explores MobileNet/EfficientNet-Lite for smartphones
   - Related Papers: #10, #30, #2

4. **Missing Explainability (XAI)**
   - All CNN/YOLO papers are black-box models
   - None uses Grad-CAM, SHAP, or LIME
   - Critical for user trust in automated sorting
   - Related Papers: #30, #10, #26

5. **No IoT + Vision Integration for Smart Bins**
   - Paper #10 proposes IoT architecture but doesn't implement it
   - Paper #5 uses blockchain for medical waste but no vision
   - No fully implemented end-to-end system
   - Related Papers: #10, #5, #26

### Task 3: 30% Implementation (DONE)
Completed in Google Colab with the real Kaggle dataset:

- Dataset downloaded and extracted (25,077 images)
- Folder structure: DATASET/TRAIN/{O, R} and DATASET/TEST/{O, R}
- O = Organic, R = Recyclable
- Sample images visualized (12 images, 6 per class)
- Class distribution analyzed (pie chart + bar chart)
- Image size analysis done (width/height histograms)
- Data preprocessing pipeline:
  - Resize to 128x128
  - Data augmentation: RandomHorizontalFlip(0.5), RandomRotation(15), ColorJitter
  - Normalization: ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Train/Val/Test split: 80% train, 20% val (from train folder), separate test folder
- DataLoaders created with batch_size=32

### Task 4: 50% Implementation (DONE)
Completed in Google Colab:

**Model Architecture — 5-layer CNN (WasteClassifierCNN):**
```python
class WasteClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3->32, 128->64
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            # Block 2: 32->64, 64->32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            # Block 3: 64->128, 32->16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            # Block 4: 128->256, 16->8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2),
            # Block 5: 256->512, 8->4
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*4*4, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))
```

**Training Configuration:**
- Epochs: 10
- Learning Rate: 0.001
- Optimizer: Adam (weight_decay=1e-4)
- Scheduler: StepLR (step_size=4, gamma=0.5)
- Loss Function: CrossEntropyLoss
- Batch Size: 32
- Device: GPU (T4 on Google Colab)
- Total Parameters: ~3.6 million
- Model Size: ~14 MB

**Results (on real Kaggle dataset):**
- Model was trained successfully for 10 epochs
- Training curves show loss decreasing and accuracy increasing
- Test evaluation done with accuracy, precision, recall, F1
- Confusion matrix generated
- Predictions visualized on random test images
- Trained model saved as best_model.pth

**Output Charts Generated (all downloaded to user's computer):**
1. 01_sample_images.png — 12 sample images from dataset
2. 02_class_distribution.png — pie chart + bar chart
3. 03_image_sizes.png — width/height distribution histograms
4. 04_augmentation.png — same image with 5 different augmentations
5. 05_training_curves.png — loss and accuracy curves over epochs
6. 06_confusion_matrix.png — 2x2 confusion matrix heatmap
7. 07_predictions.png — 10 test images with true vs predicted labels

---

## DELIVERABLES CREATED

### 1. Google Colab Notebook (.ipynb) — DONE & RUN
- File: PBL_Waste_Classification.ipynb
- 17 steps/blocks covering dataset upload to final evaluation
- User has already run this on Colab with the real Kaggle dataset
- All outputs generated successfully
- All chart files downloaded

### 2. Word Report (.docx) — DONE
- File: PBL_Report_Waste_Classification.docx
- 10 pages with:
  - Title page
  - Table of contents
  - Introduction
  - Datasets (with class distribution chart)
  - Research gap analysis (5 gaps with visualization)
  - 30% Implementation (preprocessing pipeline, dataset stats table)
  - 50% Implementation (model architecture, training config table, training curves)
  - Results (test metrics table, confusion matrix, comparison table)
  - Conclusion
  - References (8 papers cited)

### 3. PowerPoint Presentation (.pptx) — DONE
- File: PBL_Presentation.pptx
- 15 slides:
  1. Title slide (blue background)
  2. Agenda (8 items)
  3. Problem statement & objectives
  4. Datasets used (two cards side by side)
  5. Dataset analysis (class distribution chart)
  6. Research gaps identified (5 gaps with color-coded bars)
  7. 30% Implementation — preprocessing pipeline (5 step boxes)
  8. 50% Implementation — CNN architecture (architecture diagram)
  9. Training results (training curves chart)
  10. Test set evaluation (confusion matrix + metrics)
  11. Model predictions on test images
  12. Comparison with published results (table)
  13. Live demo — web application (Gradio mockup)
  14. Conclusion & future work
  15. Thank you slide

### 4. Frontend Demo (Gradio Web App) — CODE READY
- File: gradio_demo_clean.py
- One cell to add in Colab notebook
- Creates a live web interface where:
  - User uploads any waste image
  - Trained CNN model predicts Organic or Recyclable
  - Shows confidence score
  - Generates a public URL (share=True) valid for 72 hours
- User attempted to run but got SyntaxError (Unicode character issue)
- Fixed version provided as gradio_demo_clean.py (pure ASCII)
- User has NOT yet successfully run this

---

## WHAT STILL NEEDS TO BE DONE

1. **Gradio Demo:** User needs to run the clean gradio_demo_clean.py in Colab
   - Delete the old cell with the error
   - Add new cell with clean code
   - Run it to get the public URL
   - Test by uploading waste images

2. **Optional improvements for higher marks:**
   - Add more evaluation metrics (ROC curve, per-class accuracy)
   - Add Grad-CAM visualization (addresses Research Gap #4)
   - Try transfer learning with ResNet18 or MobileNetV2 for comparison
   - Add more training epochs or fine-tuning

---

## TECHNICAL DETAILS

### Dataset Structure on Google Colab
```
/content/dataset/dataset/DATASET/
    TRAIN/
        O/     (Organic images — .jpg files)
        R/     (Recyclable images — .jpg files)
    TEST/
        O/     (Organic images — .jpg files)
        R/     (Recyclable images — .jpg files)
```

### Key Variable: DATASET_DIR
```python
DATASET_DIR = "/content/dataset/dataset/DATASET"
train_dir = os.path.join(DATASET_DIR, "TRAIN")
test_dir = os.path.join(DATASET_DIR, "TEST")
```

### Libraries Used
- Python 3.x
- PyTorch (torch, torchvision)
- NumPy
- Matplotlib
- Seaborn
- Pillow (PIL)
- scikit-learn (metrics)
- Gradio (for web demo)

### Model Input/Output
- Input: RGB image resized to 128x128 pixels, normalized
- Output: 2 classes — index 0 = Organic (folder O), index 1 = Recyclable (folder R)
- Confidence: softmax probabilities

### References
1. Nnamoko, N. et al. (2022). Infrastructures, 7(4), 47. https://www.mdpi.com/2412-3811/7/4/47
2. Rai, S. et al. (2026). Applied Sciences, 16(4), 2152. https://www.mdpi.com/2076-3417/16/4/2152
3. Prakash, U. (2026). Automation, 4(2), 16.
4. Jaglan, A.K. (2022). Sustainability, 14(14), 8361.
5. Sahoo, K.C. (2022). Int. J. Environ. Res. Public Health, 19(12), 7321.
6. Mohamed, N.H. (2026). Sustainability, 18(9), 4558.
7. Manakkakudy, A. (2024). Sensors, 24(3), 809.
8. Kerboua, K. (2026). Circular Economy and Sustainability, 42(1), 2.
