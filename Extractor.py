import os
import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

# Función para procesar una sola imagen y extraer características
def procesar_imagen(file_path):
    try:
        img = cv2.imread(file_path)
        if img is None:
            return None
        img = cv2.resize(img, (128, 128))
        
        # Extracción de características (histograma de color)
        color_hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
        
        return color_hist
    except Exception as e:
        return None

# Función para paralelizar el procesamiento de imágenes
def procesar_dataset_en_paralelo(image_paths):
    with Pool(cpu_count()) as pool:
        features = pool.map(procesar_imagen, image_paths)
    return [f for f in features if f is not None]

# Ejecución principal
if __name__ == "__main__":
    # Directorio con las imágenes
    image_dir = 'dataset/dataset_updated/training_set'
    etiquetas = []
    features = []

    # Recorremos las carpetas de clases
    for label in os.listdir(image_dir):
        class_dir = os.path.join(image_dir, label)
        if not os.path.isdir(class_dir):
            continue
        image_paths = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.lower().endswith(('.jpg', '.jpeg'))]
        
        # Procesamos en paralelo y extendemos los resultados
        class_features = procesar_dataset_en_paralelo(image_paths)
        features.extend(class_features)
        etiquetas.extend([label] * len(class_features))

    # Guardamos en Excel
    df = pd.DataFrame(features)
    df['label'] = etiquetas
    df.to_excel('art_features.xlsx', index=False)
