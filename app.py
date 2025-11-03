import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import platform

# ⚙️ Antes de importar keras, definimos el backend para evitar que busque TensorFlow
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras.models import load_model  # ahora sí, usa Torch backend

# Mostrar versión de Python
st.write("Versión de Python:", platform.python_version())

# Forzar backend Torch explícitamente
keras.backend.set_backend("torch")

# Cargar el modelo entrenado
model = load_model("keras_model.h5", compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("🎶 Taylor Vision - Clasificador de Imágenes")
st.markdown("""
Convierte tu cámara en una herramienta de detección inspirada en las eras de Taylor.  
El modelo reconocerá tus poses y gestos al estilo *Fearless* o *Red* 💃
""")

image = Image.open("OIG5.jpg")
st.image(image, width=350, caption="Pose Like Taylor ✨")

with st.sidebar:
    st.subheader("Sobre esta app")
    st.markdown("""
    Entrenada con **Teachable Machine**, esta IA identifica  
    distintas posiciones en imágenes capturadas con tu cámara.
    """)

# Capturar foto
img_file_buffer = st.camera_input("📸 Toma una foto y deja que Taylor Vision la interprete")

if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    img = img.resize((224, 224))
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Predicción
    prediction = model.predict(data)
    if prediction[0][0] > 0.5:
        st.header(f"💫 Movimiento tipo *Left Era* con probabilidad {prediction[0][0]:.2f}")
    if prediction[0][1] > 0.5:
        st.header(f"🎤 Movimiento tipo *Fearless Pose* con probabilidad {prediction[0][1]:.2f}")

st.caption("📸 Desarrollado por Migue — powered by Keras & Torch ✨")
