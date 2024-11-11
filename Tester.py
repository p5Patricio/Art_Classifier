import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
# Usa rutas absolutas en lugar de relativas
import os



# Cargar el modelo, LabelEncoder y scaler
model = load_model("modelo_clasificador_arte.h5")
with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Función para extraer características de una imagen individual
def extraer_caracteristicas_imagen(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (128, 128))
    
    # Extracción de características (histograma de color)
    color_hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    return color_hist

# Función para clasificar una imagen individual
def clasificar_imagen(file_path):
    # Extraer características
    # Obtén la ruta absoluta
    absolute_path = os.path.abspath("dataset/dataset_updated/validation_set/iconography/600.jpg")

    # Verifica si el archivo existe
    if not os.path.exists(absolute_path):
        print(f"El archivo no existe en la ruta: {absolute_path}")
    features = extraer_caracteristicas_imagen(file_path)
    
    # Escalar características
    features = scaler.transform([features])
    
    # Realizar predicción
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    
    print(f"La imagen {file_path} fue clasificada como: {predicted_label[0]}")

# Clasificar una imagen de prueba
clasificar_imagen(r"dataset\dataset_updated\training_set\iconography\prueba.jpg")