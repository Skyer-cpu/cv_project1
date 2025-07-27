import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageFilter
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# --- Конфигурация ---
MODEL_DIR = "models"
IMAGE_DIR = "images"
MODEL_NAME = "best_face.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# --- Инициализация страницы ---
st.set_page_config(page_title="Face Detection App", layout="wide")
st.title("Face Detection and Blurring")
st.write("Upload an image to detect and blur faces")

# --- Вспомогательные функции ---
def validate_image(image):
    """Проверяет, является ли объект допустимым изображением"""
    return isinstance(image, (Image.Image, np.ndarray))

def display_image(image, caption):
    """Безопасное отображение изображения"""
    if validate_image(image):
        # Убрано use_container_width для совместимости
        st.image(image, caption=caption)
        return True
    st.error("Invalid image format")
    return False

def safe_image_display(image_path, caption):
    """Загрузка и отображение изображения из файла"""
    try:
        if os.path.exists(image_path):
            img = Image.open(image_path)
            return display_image(img, caption)
        st.warning(f"Image not found: {image_path}")
        return False
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return False

# --- Загрузка модели ---
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        return None
    
    try:
        model = YOLO(path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

# --- Основная функция обработки ---
def process_image(input_image, model):
    if model is None:
        return None
    
    try:
        # Конвертируем в RGB если нужно
        if isinstance(input_image, np.ndarray):
            img_pil = Image.fromarray(input_image).convert('RGB')
        else:
            img_pil = input_image.convert('RGB')
            
        img_np = np.array(img_pil)
        results = model(img_np)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_pil.width, x2), min(img_pil.height, y2)
                
                face = img_pil.crop((x1, y1, x2, y2))
                blurred = face.filter(ImageFilter.GaussianBlur(20))
                img_pil.paste(blurred, (x1, y1))
        
        return img_pil
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None

# --- Отображение метрик ---
def show_f1_analysis():
    st.subheader("F1-Confidence Curve")
    if safe_image_display(os.path.join(IMAGE_DIR, "BoxF1_curveSP.png"), ""):
        st.markdown("""
        - Peak F1=0.85 at confidence=0.33
        - Optimal range: 0.3-0.4
        """)

def show_precision_analysis():
    st.subheader("Precision Curve")
    if safe_image_display(os.path.join(IMAGE_DIR, "BoxP_curveSP.png"), ""):
        st.markdown("""
        - Max precision 1.0 at confidence=0.93
        - Recommended range: 0.6-0.8
        """)

def show_recall_analysis():
    st.subheader("Recall Curve")
    if safe_image_display(os.path.join(IMAGE_DIR, "BoxR_curveSP.png"), ""):
        st.markdown("""
        - Max recall 0.93 at confidence=0
        - For screening: confidence < 0.2
        """)

def show_pr_curve():
    st.subheader("PR Curve")
    if safe_image_display(os.path.join(IMAGE_DIR, "BoxPR_curveSP.png"), ""):
        st.markdown("""
        - mAP@0.5 = 0.879
        - Good performance up to Recall=0.85
        """)

def show_confusion_matrices():
    st.subheader("Confusion Matrices")
    cols = st.columns(2)
    with cols[0]:
        safe_image_display(os.path.join(IMAGE_DIR, "confusion_matrixSP.png"), "Confusion Matrix")
    with cols[1]:
        safe_image_display(os.path.join(IMAGE_DIR, "confusion_matrix_normalizedSP.png"), "Normalized Matrix")

def show_map_metrics():
    st.subheader("Training Progress")
    try:
        epochs = list(range(1, 21))
        map50 = [0.775, 0.793, 0.798, 0.816, 0.828, 0.838, 0.844, 0.848, 0.853, 0.857,
                 0.857, 0.863, 0.862, 0.867, 0.868, 0.871, 0.874, 0.875, 0.877, 0.879]
        map50_95 = [0.464, 0.482, 0.504, 0.517, 0.525, 0.537, 0.551, 0.549, 0.558, 0.560,
                    0.561, 0.564, 0.564, 0.568, 0.575, 0.578, 0.582, 0.580, 0.585, 0.586]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, map50, label='mAP50')
        ax.plot(epochs, map50_95, label='mAP50-95')
        ax.set_title('Training Metrics')
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting metrics: {str(e)}")

# --- Основной интерфейс ---
def main():
    model = load_model(MODEL_PATH)
    
    # Загрузка изображения
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            input_image = Image.open(uploaded_file)
            display_image(input_image, "Original Image")
            
            if model:
                with st.spinner("Processing..."):
                    result = process_image(input_image, model)
                
                if result:
                    display_image(result, "Processed Image")
                    
                    # Кнопка скачивания
                    buf = BytesIO()
                    result.save(buf, format="PNG")
                    st.download_button(
                        "Download Result",
                        buf.getvalue(),
                        "blurred_faces.png",
                        "image/png"
                    )
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    
    # Отображение метрик
    st.header("Model Performance")
    show_f1_analysis()
    show_precision_analysis()
    show_recall_analysis()
    show_pr_curve()
    show_confusion_matrices()
    show_map_metrics()

if __name__ == "__main__":
    main()