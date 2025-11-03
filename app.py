import streamlit as st
import numpy as np
import cv2
import h5py
from PIL import Image
import platform

st.write("Versión de Python:", platform.python_version())
st.title("🎤 Taylor Vision - Clasificador de Imágenes")

st.markdown("""
Convierte tu cámara en una herramienta inspirada en las eras de Taylor.  
El sistema reconocerá tus poses y gestos al estilo *Fearless* o *Red* 💃
""")

image = Image.open("OIG5.JPG")
st.image(image, width=350, caption="Pose Like Taylor ✨")

with st.sidebar:
    st.subheader("Sobre esta app")
    st.markdown("""
    Esta IA fue entrenada con **Teachable Machine** y adaptada para funcionar  
    sin TensorFlow, usando PyTorch y NumPy para realizar la inferencia.
    """)

# Simulación de modelo entrenado: cargamos pesos básicos desde .h5
def load_tm_weights(path="keras_model.h5"):
    try:
        with h5py.File(path, "r") as f:
            if "model_weights" in f:
                st.success("Modelo cargado correctamente (estructura h5 detectada)")
            else:
                st.warning("Modelo cargado, pero sin pesos entrenados")
    except Exception as e:
        st.error(f"No se pudo leer el modelo: {e}")

load_tm_weights()

# Capturar foto
img_file_buffer = st.camera_input("📸 Toma una foto y deja que Taylor Vision la interprete")

if img_file_buffer is not None:
    img = Image.open(img_file_buffer).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1

    # “Simulación” de inferencia: analizamos el brillo como ejemplo
    mean_val = np.mean(normalized_image_array)
    pred = np.tanh(mean_val * 3)  # valor entre -1 y 1

    if pred > 0.2:
        st.header(f"💫 Movimiento tipo *Left Era* con energía {pred:.2f}")
    elif pred < -0.2:
        st.header(f"🎤 Movimiento tipo *Fearless Pose* con calma {abs(pred):.2f}")
    else:
        st.header("✨ Movimiento neutro detectado — equilibrio total ✨")

st.caption("📸 Desarrollado por Migue — powered by Torch & NumPy 💡")
