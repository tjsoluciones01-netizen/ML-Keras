# Proyecto de Clasificación "Piedra, Papel o Tijera"

Este proyecto implementa una aplicación interactiva utilizando Streamlit y un modelo de Deep Learning (Keras/TensorFlow) para clasificar imágenes de manos haciendo los gestos de Piedra, Papel o Tijera. Está diseñado bajo las mejores prácticas del ciclo de vida de Machine Learning para un Ingeniero de Datos y Especialista en Machine Learning.

## 📌 El Problema
En el desarrollo de sistemas de interacción humano-computadora, el reconocimiento de gestos es un desafío fundamental. El clásico juego de "Piedra, Papel o Tijera" sirve como un excelente caso de estudio para entrenar modelos de visión computacional que puedan interpretar comandos visuales en tiempo real.

## 🎯 El Objetivo
Desplegar un modelo de Machine Learning pre-entrenado (formato `.h5`) a través de una interfaz gráfica amigable que permita a los usuarios interactuar con el modelo ya sea subiendo una imagen o utilizando su cámara web, acercando la inteligencia artificial al usuario final.

## 🔄 Ciclo de Vida del Proyecto

1. **Business Understanding (Entendimiento del Negocio):** Definición del problema de reconocimiento de gestos.
2. **Data Acquisition (Adquisición de Datos):** Recolección de imágenes representativas para "Piedra", "Papel" y "Tijera" (fase previa a este despliegue).
3. **Modeling (Modelado):** Entrenamiento de un modelo de red neuronal convolucional (exportado desde Teachable Machine o similar) para clasificar las imágenes.
4. **Deployment (Despliegue):** Esta aplicación Streamlit, que expone el modelo para su consumo final y uso práctico.

## 🛠️ Tecnologías y Requisitos

- Python 3.8+
- Streamlit
- TensorFlow / Keras
- NumPy
- Pillow (PIL)

## 🚀 Cómo Ejecutar el Proyecto

1. Clona o descarga este repositorio/carpeta.
2. Asegúrate de tener Python instalado.
3. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecuta la aplicación de Streamlit:
   ```bash
   streamlit run app.py
   ```
5. Abre el enlace local proporcionado (usualmente `http://localhost:8501`) en tu navegador web.

## 👤 Desarrollado por
**@iacam360** - Data Engineer & Machine Learning Specialist.
