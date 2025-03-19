from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import os  


app = FastAPI()

origins = [
    "http://localhost:4200",          # Geliştirme ortamı
    "https://hasanbarisgok.com",        # Üretim ortamı (www olmadan)
    "https://www.hasanbarisgok.com",    # Üretim ortamı (www ile)
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tüm domainlere izin ver
    allow_credentials=False,      # Credentials kullanılmıyorsa False yapabilirsiniz
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "covid19Model.onnx"
class_names = ['COVID19', 'NORMAL', 'PNEUMONIA']

# Uygulama başlatıldığında modeli yükle
ort_session = ort.InferenceSession(MODEL_PATH)

def preprocess_image(image_bytes: bytes):
    # Görseli gri tonlamalı olarak aç
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    # 224x224 boyutuna getir
    img = img.resize((224, 224))
    # Numpy array'e çevir (normalizasyon yok)
    img_array = np.array(img).astype(np.float32)
    # ONNX modelinin beklediği formata getir (NCHW: Batch x Channels x Height x Width)
    img_array = img_array.reshape(1, 1, 224, 224)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Sadece JPEG ve PNG dosya türlerini kabul et
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Desteklenmeyen dosya türü. Lütfen JPEG veya PNG kullanın.")
    
    # Dosyayı oku
    image_bytes = await file.read()
    input_image = preprocess_image(image_bytes)
    
    # Modelin giriş adını al
    input_name = ort_session.get_inputs()[0].name
    # Tahmin yap
    outputs = ort_session.run(None, {input_name: input_image})
    probabilities = outputs[0][0]
    
    # En yüksek olasılığa sahip sınıfı belirle
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(probabilities))
    
    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": {cls: float(prob) for cls, prob in zip(class_names, probabilities)}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))  # Use Heroku's assigned port
    )