import numpy as np
from PIL import Image
import onnxruntime as ort

def preprocess_image(image_path):
    # Görseli gri tonlamalı olarak aç
    img = Image.open(image_path).convert('L')
    
    # 224x224 boyutuna getir
    img = img.resize((224, 224))
    
    # Numpy array'e çevir (normalizasyon YOK)
    img_array = np.array(img).astype(np.float32)
    
    # ONNX modelinin beklediği formata getir (NCHW)
    img_array = img_array.reshape(1, 1, 224, 224)  # Batch x Channels x Height x Width
    
    return img_array

def predict(image_path, model_path):
    # Önişleme
    input_image = preprocess_image(image_path)
    
    # ONNX modelini yükle
    ort_session = ort.InferenceSession(model_path)
    
    # Giriş adını kontrol et
    input_name = ort_session.get_inputs()[0].name
    
    # Tahmin yap
    outputs = ort_session.run(None, {input_name: input_image})
    
    # Sınıf olasılıklarını al (3 sınıf)
    probabilities = outputs[0][0]
    
    return probabilities

if __name__ == "__main__":
    # Kullanıcıdan giriş alınır.
    image_path = "v1.png"
    
    # Eğer model yolu boş bırakılırsa varsayılan modeli kullan
    
    model_path = "covid19Model.onnx"
    
    # Sınıf isimleri (MATLAB'deki sırayla aynı olmalı)
    class_names = ['COVID19', 'NORMAL', 'PNEUMONIA'] 
    
    # Tahminleri al
    probabilities = predict(image_path, model_path)
    predicted_class = class_names[np.argmax(probabilities)]
    confidence = np.max(probabilities)
    
    print(f"\nPrediction: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("\nClass Probabilities:")
    for cls, prob in zip(class_names, probabilities):
        print(f"{cls}: {prob:.4f}")
