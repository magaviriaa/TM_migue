import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# Mostrar la versión del sistema
st.write("Versión de Python:", platform.python_version())

# Cargar modelo de Teachable Machine
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Interfaz inspirada
st.title("✨ Taylor Vision 🎶")
st.markdown("""
Convierte tu cámara en un detector de movimientos al estilo de un videoclip de Taylor Swift.  
El modelo reconocerá tus poses y gestos en tiempo real 💃📸
""")

# Imagen de portada
image = Image.open('OIG5.jpg')
st.image(image, width=350, caption="Pose Like Taylor ✨")

with st.sidebar:
    st.subheader("Sobre esta app")
    st.markdown("""
    Esta cámara utiliza un modelo de **Teachable Machine**  
    para reconocer posiciones básicas y clasificarlas.  
    ¡Pruébala e imagina que estás grabando tu propia era! 💫
    """)

# Captura desde la cámara
img_file_buffer = st.camera_input("Toma una foto y deja que Taylor Vision la interprete 💁‍♀️")

if img_file_buffer is not None:
    # Convertir la imagen a un array compatible con el modelo
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)
    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    # Normalización
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Inferencia
    prediction = model.predict(data)
    print(prediction)

    # Interpretar resultados
    if prediction[0][0] > 0.5:
        st.header(f"💫 Movimiento tipo *Left Era* con probabilidad {prediction[0][0]:.2f}")
    if prediction[0][1] > 0.5:
        st.header(f"🎤 Movimiento tipo *Fearless Pose* con probabilidad {prediction[0][1]:.2f}")
    #if prediction[0][2] > 0.5:
    #    st.header(f"🔥 Movimiento tipo *Right Beat* con probabilidad {prediction[0][2]:.2f}")

st.caption("📸 Desarrollado por Migue — powered by Teachable Machine y Taylor vibes 💖")
