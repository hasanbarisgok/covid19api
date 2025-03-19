from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from PIL import Image
import onnxruntime as ort
from io import BytesIO
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for the response
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: List[dict]

# Load the ONNX model during startup
@app.on_event("startup")
async def load_model():
    try:
        model_path = "covid19Model.onnx"
        app.state.model = ort.InferenceSession(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise RuntimeError("Model initialization failed")

def preprocess_image(image_bytes: bytes):
    try:
        img = Image.open(BytesIO(image_bytes)).convert('L')  # Convert to grayscale
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32)
        img_array = img_array.reshape(1, 1, 224, 224)  # Reshape to NCHW
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise ValueError("Invalid image file")

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    class_names = ['COVID19', 'NORMAL', 'PNEUMONIA']
    
    try:
        # Read and preprocess the image
        image_data = await file.read()
        input_tensor = preprocess_image(image_data)
        
        # Run inference
        ort_session = app.state.model
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: input_tensor})
        probabilities = outputs[0][0]
        
        # Generate response
        predicted_idx = np.argmax(probabilities)
        response = {
            "prediction": class_names[predicted_idx],
            "confidence": float(probabilities[predicted_idx]),
            "probabilities": [
                {"class": cls, "probability": float(prob)}
                for cls, prob in zip(class_names, probabilities)
            ]
        }
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "API is running"}