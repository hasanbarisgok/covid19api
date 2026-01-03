# COVID-19 & Pneumonia Detection from Chest X-Ray (CXR)
**End-to-End Medical AI System — MATLAB (Training) → ONNX (Interoperability) → FastAPI (Serving) → Cloud Deployment**

This repository provides a lightweight **FastAPI** inference service that classifies **Chest X-Ray (CXR)** images into:
- **COVID19**
- **NORMAL**
- **PNEUMONIA**

The model was trained in **MATLAB** as a custom CNN, exported to **ONNX** for portability, and served via **ONNX Runtime** in a production-like API setup (proof-of-concept full-stack deployment).

> ⚠️ **Disclaimer (Research Only):** This project is a proof-of-concept for academic/research purposes and **must not** be used for clinical diagnosis or medical decision-making.

---

## Table of Contents
- [Key Features](#key-features)
- [System Overview](#system-overview)
- [Model & Data Summary](#model--data-summary)
- [API](#api)
- [Quickstart (Local)](#quickstart-local)
- [Example Request](#example-request)
- [Response Schema](#response-schema)
- [Deployment Notes](#deployment-notes)
- [Project Structure](#project-structure)
- [Limitations & Ethics](#limitations--ethics)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Data Source and Licensing
**Primary dataset source:** Kaggle

- Dataset: **Chest X-Ray (COVID19 & Pneumonia)**  
  https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia

- The dataset was **open/publicly accessible** at the time of development.
- This repository does **not** redistribute any dataset archives.
- Only a sample image (e.g., `example.jpg` / `example.png`) may be included for API testing.

> If you plan to publish this repository publicly, keep the dataset link above and mention the dataset terms/license in `README.md`.




## Key Features
- **FastAPI** backend with a single `/predict` endpoint for image classification
- **ONNX Runtime** inference (portable, efficient)
- Simple preprocessing pipeline (**grayscale → resize 224×224 → NCHW tensor**)
- CORS enabled for easy frontend integration
- Cloud-friendly files included (e.g., `Procfile`, `runtime.txt`)

---

## System Overview
High-level pipeline:

1. **Upload CXR image** (JPG/PNG)
2. **Preprocess**
   - Convert to **grayscale**
   - Resize to **224×224**
   - Convert to float32 tensor shaped as **(1, 1, 224, 224)** (NCHW)
3. **Run ONNX inference**
4. **Return** predicted class + confidence + per-class probabilities

---

## Model & Data Summary

### Classes
- `COVID19`, `NORMAL`, `PNEUMONIA`

### Dataset (high-level)
The dataset was constructed by combining:
- Kaggle **Chest X-Ray Images (Pneumonia)**
- Public COVID-19 CXR archives

A balanced distribution was targeted across classes.

> If you publish this repo publicly, ensure you have the right to redistribute any sample images and clearly document the dataset sources and licenses.

### Preprocessing (training & inference aligned)
- Resize to **224×224**
- Input range: [0, 255] (no scaling required
- Convert to grayscale (single-channel)

### Training (summary)
- Custom CNN trained in **MATLAB Deep Learning Toolbox**
- Exported to ONNX for deployment and interoperability

### Evaluation (recommended to include)
If you have metrics from your experiments, add them here (Accuracy / Precision / Recall / F1, plus a confusion matrix figure).
This strengthens the repository as academic evidence.

Example table format:
| Class       | Precision | Recall | F1 |
|------------|-----------|--------|----|
| COVID19    |           |        |    |
| NORMAL     |           |        |    |
| PNEUMONIA  |           |        |    |
| **Macro-F1** |         |        |    |

---

## API

### Base URL
- Local: `http://127.0.0.1:8000`

### Endpoints

#### `GET /`
Simple health check.

**Response**
```json
{ "status": "API is running" }
```

#### `POST /predict`
Predicts the class from an uploaded CXR image.

- **Content-Type:** `multipart/form-data`
- **Form field:** `file` (image)

**Returns**
- `prediction`: string (one of `COVID19`, `NORMAL`, `PNEUMONIA`)
- `confidence`: float (probability of predicted class)
- `probabilities`: list of `{class, probability}` for all classes

---

## Quickstart (Local)

### Requirements
- Python 3.9+ (recommended)
- pip

### Install
```bash
pip install -r requirements.txt
```

### Run the API
Assuming your file is named `api.py`:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

You should see logs like:
- `Model loaded successfully`

---

## Example Request

### cURL
```bash
curl -X POST "http://127.0.0.1:8000/predict"   -H "accept: application/json"   -F "file=@example.jpg"
```

### Python (requests)
```python
import requests

url = "http://127.0.0.1:8000/predict"
with open("example.jpg", "rb") as f:
    files = {"file": ("example.jpg", f, "image/jpeg")}
    r = requests.post(url, files=files)

print(r.json())
```

---

## Response Schema
Example response:
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.9361,
  "probabilities": [
    {"class": "COVID19", "probability": 0.0123},
    {"class": "NORMAL", "probability": 0.0516},
    {"class": "PNEUMONIA", "probability": 0.9361}
  ]
}
```

---

## Deployment Notes
This repo includes deployment-related files commonly used for PaaS setups:
- `Procfile` (for process definition)
- `runtime.txt` (runtime hint)
- `requirements.txt`

Typical deployment pattern:
- **Backend** on a PaaS provider (e.g., Heroku-style deployment)
- **Frontend** on a static hosting platform (e.g., Vercel)

> If you plan to deploy publicly, consider:
- removing any sample medical images if licensing is unclear
- adding rate limiting / request size limits
- adding input validation (file type, max size)
- enabling structured logging & monitoring

---

## Project Structure
```text
.
├── api.py                 # FastAPI app (inference server)
├── covid19Model.onnx       # ONNX model file
├── requirements.txt
├── runtime.txt
├── Procfile
├── example.jpg            # sample request image (optional)
└── deneme.py              # experiments / local tests (optional)
```

---

## Limitations & Ethics
- **Not a medical device.** This is an academic prototype.
- **Data bias risk:** performance may not generalize to different hospitals, devices, demographics, or acquisition protocols.
- **Privacy & compliance:** do not upload real patient data unless you have proper authorization and compliance (KVKK/GDPR).
- **Explainability:** CNN-based decisions are not inherently interpretable; future work may include Grad-CAM/SHAP-style explanations.

---

## License
Specify a license before public release (e.g., MIT/Apache-2.0).  
If you do not want reuse, consider “All rights reserved”.

---

## Contributors & Roles

This project was developed collaboratively. Roles are summarized below for transparency:

- **Hasan Barış GÖK**
   - Supported the **model export handoff** used for ONNX deployment (model artifact provided)
  - Built the **FastAPI inference service** (endpoints, request/response schema)
  - Implemented **ONNX Runtime integration** and **image preprocessing** pipeline
  - Managed **deployment configuration** (e.g., Procfile/runtime) and overall repository structure
  - Prepared documentation (README, provenance notes)

- **Ahmet Nihat VELİOĞLU**
  - Developed/trained the **MATLAB model** (CNN training workflow)
