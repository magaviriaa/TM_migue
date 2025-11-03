import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# Mostrar versión de Python
st.write("Versión de Python:", platform.python_version())

# Cargar el modelo entrenado
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Título y narrativa Taylor
st.title("✨ Taylor Vision 🎶")
st.markdown("""
Convierte tu cámara en un detector de poses al estilo de los videoclips de Taylor Swift.  
El modelo reconocerá tus gestos y movimientos en tiempo real 💃📸
""")

# Imagen de portada
image = Image.open('OIG5.jpg')
st.image(image, width=350, caption="Pose Like Taylor ✨")

# Barra lateral
with st.sidebar:
    st.subheader("Sobre esta app")
    st.markdown("""
    Esta cámara usa un modelo de **Teachable Machine**  
    para identificar posiciones o movimientos.  
    ¡Imagina que estás grabando tu propia era! 💫
    """)

# Captura desde la cámara
img_file_buffer = st.camera_input("📸 Toma una foto y deja que Taylor Vision la interprete")

if img_file_buffer is not None:
    # Preparar imagen para el modelo
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)
    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    # Normalizar
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Predicción
    prediction = model.predict(data)
    print(prediction)

    # Mostrar resultados
    if prediction[0][0] > 0.5:
        st.header(f"💫 Movimiento tipo *Left Era* con probabilidad {prediction[0][0]:.2f}")
    if prediction[0][1] > 0.5:
        st.header(f"🎤 Movimiento tipo *Fearless Pose* con probabilidad {prediction[0][1]:.2f}")
    # if prediction[0][2] > 0.5:
    #     st.header(f"🔥 Movimiento tipo *Right Beat* con probabilidad {prediction[0][2]:.2f}")

st.caption("📸 Desarrollado por Migue — powered by Teachable Machine & Taylor vibes 💖")
