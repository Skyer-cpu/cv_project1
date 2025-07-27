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
MODEL_NAME = "best_face.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
IMAGE_DIR = "images"

# --- Инициализация страницы ---
st.set_page_config(page_title="Face Detection & Blurring", layout="wide")
st.title("Обнаружение и размытие лиц с помощью YOLOv8")
st.write("Загрузите изображение для обработки")

# --- Утилиты ---
def load_image(image_path):
    """Безопасная загрузка изображения"""
    try:
        return Image.open(image_path)
    except Exception as e:
        st.error(f"Ошибка загрузки изображения {image_path}: {str(e)}")
        return None

def safe_image_display(image_path, caption):
    """Отображение изображения с обработкой ошибок"""
    img = load_image(image_path)
    if img:
        st.image(img, caption=caption, use_container_width=True)
        return True
    return False

# --- Загрузка модели ---
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Модель не найдена: {path}")
        return None
    
    try:
        model = YOLO(path)
        st.success("Модель успешно загружена!")
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        return None

trained_model = load_model(MODEL_PATH)

# --- Обработка изображений ---
def blur_faces(image, model, blur_radius=20):
    if model is None:
        return image

    try:
        img_pil = image.convert('RGB')
        img_np = np.array(img_pil)
        results = model(img_np)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_pil.width, x2), min(img_pil.height, y2)
                
                face = img_pil.crop((x1, y1, x2, y2))
                blurred = face.filter(ImageFilter.GaussianBlur(blur_radius))
                img_pil.paste(blurred, (x1, y1))
        return img_pil
    except Exception as e:
        st.error(f"Ошибка обработки: {str(e)}")
        return image

# --- Основной интерфейс ---
st.header("Загрузка изображения")
uploaded_file = st.file_uploader("Выберите файл...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Исходное изображение", use_column_width=True)
    
    if trained_model:
        with st.spinner("Обработка изображения..."):
            blurred = blur_faces(input_image, trained_model)
        
        st.image(blurred, caption="Результат обработки", use_column_width=True)
        
        # Кнопка скачивания
        buf = BytesIO()
        blurred.save(buf, format="PNG")
        st.download_button(
            "Скачать результат",
            buf.getvalue(),
            "blurred_face.png",
            "image/png"
        )

# --- Визуализация метрик ---
st.header("📊 Метрики производительности модели")

def show_metric_analysis(image_name, title, analysis_text):
    st.subheader(title)
    if safe_image_display(os.path.join(IMAGE_DIR, image_name), title):
        st.markdown(analysis_text)

# F1-анализ
show_metric_analysis(
    "BoxF1_curveSP.png",
    "Анализ F1-Confidence Curve",
    """
    ### Основные наблюдения:
    - Пик F1=0.85 при confidence=0.33
    - Оптимальный диапазон: 0.3-0.4
    """
)

# Precision-анализ
show_metric_analysis(
    "BoxP_curveSP.png",
    "Анализ Precision-Confidence Curve",
    """
    ### Основные наблюдения:
    - Максимальная точность 1.0 при confidence=0.93
    - Для баланса рекомендуемый диапазон: 0.6-0.8
    """
)

# Recall-анализ
show_metric_analysis(
    "BoxR_curveSP.png",
    "Анализ Recall-Confidence Curve",
    """
    ### Основные наблюдения:
    - Максимальный recall 0.93 при confidence=0
    - Для скрининга: confidence < 0.2
    """
)

# PR-анализ
show_metric_analysis(
    "BoxPR_curveSP.png",
    "Анализ Precision-Recall Curve",
    """
    ### Основные наблюдения:
    - mAP@0.5 = 0.879
    - Хорошая работа при Recall до 0.85
    """
)

# Матрицы ошибок
st.subheader("Матрицы ошибок")
cols = st.columns(2)
with cols[0]:
    safe_image_display(os.path.join(IMAGE_DIR, "confusion_matrixSP.png"), "Матрица ошибок")
with cols[1]:
    safe_image_display(os.path.join(IMAGE_DIR, "confusion_matrix_normalizedSP.png"), "Нормализованная матрица ошибок")

# График mAP
st.subheader("Динамика метрик обучения")
try:
    epochs = list(range(1, 21))
    map50 = [0.775, 0.793, 0.798, 0.816, 0.828, 0.838, 0.844, 0.848, 0.853, 0.857,
             0.857, 0.863, 0.862, 0.867, 0.868, 0.871, 0.874, 0.875, 0.877, 0.879]
    map50_95 = [0.464, 0.482, 0.504, 0.517, 0.525, 0.537, 0.551, 0.549, 0.558, 0.560,
                0.561, 0.564, 0.564, 0.568, 0.575, 0.578, 0.582, 0.580, 0.585, 0.586]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, map50, label='mAP50', marker='o')
    ax.plot(epochs, map50_95, label='mAP50-95', marker='s')
    ax.set_title('Динамика метрик mAP')
    ax.set_xlabel('Эпохи')
    ax.set_ylabel('Значение метрики')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    st.markdown("""
    ### Ключевые выводы:
    - Лучшие показатели достигаются после 15 эпохи
    - mAP50: 0.879
    - mAP50-95: 0.586
    """)
except Exception as e:
    st.error(f"Ошибка построения графиков: {str(e)}")

# --- Подвал ---
st.markdown("---")
st.caption("Разработано с использованием Streamlit и Ultralytics YOLOv8 | © 2023")