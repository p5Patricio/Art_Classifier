🚀 Art Classifier:<br>
Este proyecto usa un dataset de kaggle (https://www.kaggle.com/datasets/thedownhill/art-images-drawings-painting-sculpture-engraving) para
implementa un clasificador de arte utilizando un Perceptrón Multicapa (MLP) 
para identificar diferentes estilos o categorías artísticas basándose en características extraídas de las imágenes.
<br>
<br>
📋 Descripción: <br>
El clasificador utiliza una arquitectura de Perceptrón Multicapa (red neuronal feed-forward multicapa) implementada con TensorFlow/Keras para clasificar obras de arte en diferentes categorías. La red neuronal está diseñada con conexiones unidireccionales (hacia adelante), donde la información fluye desde la capa de entrada, a través de una capa oculta, hasta la capa de salida. El modelo procesa características extraídas previamente de las imágenes artísticas y almacenadas en un archivo Excel.

🛠️ Necesario instalar:<br>
Para instalar las dependencias necesarias usa este comando: <br>
pip install -r Dependencias.txt <br>
<br>
Para instalar el dataset de kaggle usa este codigo: <br>
import kagglehub
<br>

path = kagglehub.dataset_download("thedownhill/art-images-drawings-painting-sculpture-engraving")
<br>
print("Path to dataset files:", path)
<br>
<br>
📁 Estructura del Proyecto

ART_CLASSIFIER/<br>
│<br>
├── dataset/<br>
│   ├── dataset_updated/<br>
│   │   ├── training_set/<br>
│   │   └── validation_set/<br>
│   └── art_features.xlsx<br>
│<br>
├── modelo_clasificador_arte.h5<br>
├── label_encoder.pkl<br>
├── scaler.pkl<br>
├── Clasificador.py<br>
├── Extractor.py<br>
├── Tester.py<br>
└── Dependencias.txt<br>
