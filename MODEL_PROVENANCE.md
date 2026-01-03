# MODEL_PROVENANCE — MATLAB Training → ONNX Export → FastAPI Inference

This document captures an audit-friendly summary of **how the model was trained in MATLAB** and how the resulting artifact was **exported and deployed** in this repository.

---

## 1) Task Definition
**Task:** Chest X-Ray (CXR) multi-class image classification  
**Classes (folder labels):**
- `COVID19`
- `NORMAL`
- `PNEUMONIA`

**Output:** class probabilities + predicted label + confidence.

---

## 2) Dataset (Source & Local Layout)

### Source
Kaggle dataset used:
- https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia

### Local folder layout (MATLAB training)
The dataset is expected to be extracted into a local folder (example):
- `datasetPath = 'Covid19'`

MATLAB loads data using:
- `imageDatastore(datasetPath, IncludeSubfolders=true, LabelSource='foldernames')`

So the **class labels are derived from subfolder names** (e.g., `Covid19/COVID19`, `Covid19/NORMAL`, `Covid19/PNEUMONIA`).

> This repository does **not** redistribute the full dataset archive. Any sample image included is for API testing only.

---

## 3) Data Split (Train / Validation / Test)
The dataset is split per class using MATLAB:

- Train: **70%**
- Validation: **15%**
- Test: **15%**
- Split mode: randomized

MATLAB code:
- `splitEachLabel(imds, 0.7, 0.15, 0.15, 'randomized')`

---

## 4) Preprocessing (Training-Time)
To align with the deployed inference pipeline, preprocessing is explicitly defined.

### Steps (MATLAB)
For each image:
1. Read image file (`imread`)
2. If RGB, convert to grayscale (`rgb2gray`)
3. Resize to **224×224** (`imresize`)
4. Cast/store as **single** in a 4D tensor

### Tensor shape (MATLAB)
Data is assembled as:
- `XTrain: 224 × 224 × 1 × N` (MATLAB convention: **SSCB** = Spatial, Spatial, Channel, Batch)

### Normalization note (important)
In the provided MATLAB script, pixel values are **not explicitly normalized** (no `/255` or `rescale` call).  
Images are resized and stored as `single`, but remain in the original intensity range typically **0–255**.

**Implication:**  
The Python API should match this preprocessing behavior (no normalization) unless you re-train or re-export with normalized inputs.

---

## 5) Model Architecture (MATLAB CNN)
Input:
- `imageInputLayer([224 224 1])`

Backbone:
- Conv(3×3, 16, padding=same) → BatchNorm → ReLU → MaxPool(2, stride=2)
- Conv(3×3, 32, padding=same) → BatchNorm → ReLU → MaxPool(2, stride=2)
- Conv(3×3, 64, padding=same) → BatchNorm → ReLU → MaxPool(2, stride=2)

Head:
- FullyConnected(3) → Softmax → ClassificationLayer

This is a compact CNN suitable for coursework-level medical AI prototyping.

---

## 6) Training Configuration (MATLAB)
Training is performed using:

- Optimizer: **Adam**
- Max epochs: **50**
- Mini-batch size: **32**
- Initial learning rate: **0.01**
- Validation: `{XValidation, YValidation}`
- Validation frequency: **5**
- Training progress visualization enabled

MATLAB call:
- `trainingOptions('adam', 'MaxEpochs', 50, 'MiniBatchSize', 32, 'InitialLearnRate', 0.01, 'ValidationData', {XValidation, YValidation}, 'ValidationFrequency', 5, ...)`

---

## 7) Evaluation (MATLAB)
After training:

- Predictions: `YPred = classify(net, XTest)`
- Accuracy:
  - `accuracy = sum(YPred == YTest) / numel(YTest)`
- Confusion matrix:
  - `confusionchart(YTest, YPred)`

A single-image local test example is also included in the MATLAB script (grayscale + resize + reshape to 4D).

---

## 8) Export to ONNX & Deployment in This Repo
The trained MATLAB model was exported to **ONNX** and included in this repository as:

- `covid19Model.onnx`

### Inference runtime (Python)
The API uses:
- **ONNX Runtime** (`onnxruntime.InferenceSession`)

### Input format alignment
- MATLAB uses **224×224×1×N (SSCB)**
- The FastAPI pipeline uses **(1, 1, 224, 224) (NCHW)**

This is the expected layout for many ONNX pipelines. If the exported model expects a different input name/shape, the API obtains the input name dynamically via:
- `ort_session.get_inputs()[0].name`

---

## 9) Contributors & Roles

This project was developed collaboratively. Roles are summarized below for transparency.

### Hasan Barış GÖK (Primary contributor)
- Built the **FastAPI inference service** (endpoints, request/response schema)
- Implemented **ONNX Runtime integration** and the **image preprocessing pipeline**
- Managed **deployment configuration** (e.g., `Procfile`, `runtime.txt`) and overall repository structure
- Supported the **model export handoff** used for ONNX deployment (model artifact provided)
- Prepared documentation (README, provenance notes)

### Ahmet Nihat VELİOĞLU
- Developed/trained the **MATLAB model** (CNN training workflow)
---

## 10) Ethical Use & Limitations
- **Not for clinical diagnosis.** Research/prototype only.
- Performance may not generalize under **domain shift** (different devices/hospitals/protocols).
- Consider adding explainability (e.g., Grad-CAM) and stronger validation for real-world use.

