import streamlit as st
import tensorflow as st_tf # using standard import
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Piedra, Papel o Tijera AI",
    page_icon="✌️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
    }
    .landing-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Caché para cargar el modelo solo una vez
@st.cache_resource
def load_keras_model():
    # Deshabilitar advertencias de compilación si las hay
    model = load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        labels = f.readlines()
    # Limpiar saltos de línea y obtener solo el texto de la etiqueta
    labels = [label.strip().split(" ", 1)[1] if " " in label else label.strip() for label in labels if label.strip()]
    return model, labels

# Barra lateral para navegación
st.sidebar.title("Navegación")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/6134/6134707.png", width=100) # Icono representativo
page = st.sidebar.radio("Ir a:", ["📖 Teoría (Landing Page)", "🚀 Aplicación (Inferencia)"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Desarrollado por:** @iacam360")

if page == "📖 Teoría (Landing Page)":
    st.markdown('<div class="landing-header"><h1>✊ ✋ ✌️ <br>Proyecto de Clasificación: Piedra, Papel o Tijera</h1></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📌 El Problema")
        st.write("""
        En el desarrollo de sistemas de interacción humano-computadora, el reconocimiento de gestos es un desafío fundamental. 
        El clásico juego de **"Piedra, Papel o Tijera"** sirve como un excelente caso de estudio para entrenar modelos de visión 
        computacional que puedan interpretar comandos visuales en tiempo real y traducirlos en acciones concretas.
        """)
        
        st.header("🎯 El Objetivo")
        st.write("""
        El objetivo principal de esta aplicación es demostrar cómo se puede desplegar un modelo de Machine Learning 
        (previamente entrenado) utilizando una interfaz web amigable. Esto permite acercar la Inteligencia Artificial al 
        usuario final, permitiéndole interactuar directamente con el modelo a través de imágenes subidas o la cámara de su dispositivo.
        """)

    with col2:
        st.header("⚙️ Funcionalidad")
        st.info("""
        Esta aplicación carga un modelo neuronal profundo (formato `.h5` de Keras/TensorFlow). Toma una imagen, la redimensiona 
        a 224x224 píxeles, la normaliza y luego extrae las características visuales para calcular las probabilidades de que la 
        imagen pertenezca a una de las tres clases: **Piedra, Papel o Tijera**.
        """)
        
        st.header("👤 Quién lo desarrolló")
        st.success("**@iacam360** - Data Engineer & Machine Learning Specialist.")
        
    st.markdown("---")
    st.header("🔄 Ciclo de Vida del Desarrollo")
    st.write("""
    Este proyecto se enmarca dentro de las mejores prácticas de un Ingeniero de Datos y Especialista en Machine Learning, abarcando las siguientes fases:
    """)
    
    # Tabla usando markdown
    st.markdown("""
    | Fase | Descripción en este Proyecto |
    | :--- | :--- |
    | **1. Business Understanding** | Definición del objetivo: Reconocer gestos manuales para el juego de Piedra, Papel o Tijera. |
    | **2. Data Acquisition** | Recolección y etiquetado de cientos de imágenes de manos realizando los gestos. |
    | **3. Modeling** | Entrenamiento de una Red Neuronal Convolucional (CNN) usando Teachable Machine / Keras. |
    | **4. Deployment** | Empaquetado del modelo y exposición a través de esta aplicación en **Streamlit**. |
    """)

elif page == "🚀 Aplicación (Inferencia)":
    st.title("🚀 Clasificador de Gestos")
    st.write("Sube una imagen o toma una foto para que el modelo identifique si es **Piedra, Papel o Tijera**.")

    model, labels = load_keras_model()

    # Opciones de entrada
    input_method = st.radio("Elige el método de entrada:", ["Subir Archivo", "Usar Cámara Web"])
    
    img_file = None
    if input_method == "Subir Archivo":
        img_file = st.file_uploader("Sube una imagen de tu mano...", type=["jpg", "jpeg", "png"])
    else:
        img_file = st.camera_input("Toma una foto de tu mano")

    if img_file is not None:
        # Mostrar la imagen
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption='Imagen cargada', use_column_width=True)
        
        st.write("⏳ Procesando la imagen...")
        
        # Preprocesamiento
        # 1. Redimensionar y recortar la imagen a 224x224
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        # 2. Convertir a array de numpy
        image_array = np.asarray(image)
        
        # 3. Normalizar la imagen
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # 4. Crear el array de datos de entrada (shape: 1, 224, 224, 3)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Inferencia
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = labels[index]
        confidence_score = prediction[0][index]

        st.markdown("---")
        st.header("✨ Resultado de la Predicción")
        
        # Diccionario de emojis para hacerlo más visual
        emojis = {"papel": "✋", "piedra": "✊", "tijera": "✌️"}
        predicted_emoji = emojis.get(class_name.lower(), "🤔")
        
        st.success(f"### {predicted_emoji} Predicción: **{class_name.capitalize()}**")
        st.info(f"**Confianza del modelo:** {confidence_score * 100:.2f}%")
        
        # Mostrar gráfica de barras de las probabilidades
        st.write("### Probabilidades por clase:")
        
        # Crear datos para el gráfico
        import pandas as pd
        probs_df = pd.DataFrame({
            'Clase': [l.capitalize() for l in labels],
            'Probabilidad': prediction[0] * 100
        })
        st.bar_chart(probs_df.set_index('Clase'))
