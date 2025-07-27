import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")
st.title("🧠 Детекция опухолей мозга (YOLOv8)")  # Изменил на YOLOv8, так как YOLOv11 не существует

# --- Конфигурация ---
MODEL_DIR = "models"
MODEL_NAME = "dinara_yolov11_best.pt"  # Убедитесь, что это действительно YOLOv8 модель
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# --- Загрузка обученной модели ---
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Ошибка: Файл модели не найден по пути {path}. Убедитесь, что '{MODEL_NAME}' находится в каталоге '{MODEL_DIR}'.")
        return None
    try:
        # Проверяем доступность GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.info(f"Используется устройство: {device.upper()}")
        
        # Загружаем модель
        model = YOLO(path)
        model.to(device)
        
        # Проверяем, что модель загрузилась
        if model is not None:
            st.success(f"Модель '{MODEL_NAME}' успешно загружена! Можно приступать к использованию.")
            return model
        else:
            st.error("Не удалось загрузить модель.")
            return None
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {str(e)}")
        return None

model = load_model(MODEL_PATH)

# ===== Загрузка изображений =====
tab1, tab2 = st.tabs(["📂 Загрузить изображение", "🌐 По ссылке"])

uploaded_files = []
image_url = ""

with tab1:
    uploaded_files = st.file_uploader("Загрузите одно или несколько изображений:", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

with tab2:
    image_url = st.text_input("Вставьте ссылку на изображение (jpeg/png):")

# Обработка загруженных файлов и URL
imgs = []
if uploaded_files:
    for file in uploaded_files:
        try:
            img = Image.open(file)
            imgs.append(img)
        except Exception as e:
            st.error(f"Ошибка при открытии файла {file.name}: {str(e)}")

if image_url:
    try:
        response = requests.get(image_url, timeout=5)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            imgs.append(img)
        else:
            st.error(f"Ошибка загрузки изображения. Код статуса: {response.status_code}")
    except Exception as e:
        st.error(f"Ошибка загрузки изображения по ссылке: {str(e)}")

# ===== Инференс и отображение =====
if model is not None and imgs:
    for img in imgs:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption="Оригинал", use_column_width=True)
        
        try:
            # YOLO предсказание
            results = model.predict(img, conf=0.25)
            
            with col2:
                for r in results:
                    # Проверяем, есть ли аннотированное изображение
                    if hasattr(r, 'plot'):
                        annotated_img = r.plot()  # изображение с bbox
                        st.image(annotated_img, caption="Результат детекции", use_column_width=True)
                    else:
                        st.warning("Модель не вернула аннотированное изображение.")
                    
                    # Показываем информацию о детекции
                    if len(r.boxes) > 0:
                        st.info(f"Обнаружено {len(r.boxes)} опухолей.")
                    else:
                        st.info("Опухоли не обнаружены.")
        except Exception as e:
            st.error(f"Ошибка при обработке изображения: {str(e)}")

# ===== Информация о модели =====
with st.expander("ℹ️ Информация о модели и метрики"):
    st.markdown("""
    ### **YOLOv8** использовалась для детекции опухолей на MR-снимках.

    - 📦 **Размер обучающей выборки:** 310 изображений  
    - 📦 **Размер валидационной выборки:** 75 изображений  
    - 🔁 **Эпохи обучения:** 50 эпох *3 модели = 150  
    - 🎯 **mAP50:** 0.88  
    - 🎯 **mAP50-95:** 0.60
    """)

    # Функция для безопасного отображения изображений
    def safe_image_display(image_path, caption):
        try:
            if os.path.exists(image_path):
                img = Image.open(image_path)
                st.image(img, caption=caption, use_container_width=True)
            else:
                st.warning(f"Изображение не найдено: {image_path}")
        except Exception as e:
            st.error(f"Ошибка при загрузке изображения {image_path}: {str(e)}")

    # Отображение графиков
    st.markdown("**ГРАФИК №1: mAP - детекция опухолей мозга (YOLOv8):**")
    safe_image_display("images/dinara_map.jpg", "Confusion Matrix")
    
    st.markdown("**ГРАФИК №2 box_f1_curve - детекция опухолей мозга (YOLOv8):**")
    safe_image_display("images/dinara_BoxF1_curve.png", "F1 Curve")
    
    st.markdown("**ГРАФИК №3 PR-кривая - детекция опухолей мозга (YOLOv8):**")
    safe_image_display("images/dinara_BoxPR_curve.png", "PR Curve")
    
    st.markdown("**ГРАФИК №4 Матрица ошибок - детекция опухолей мозга (YOLOv8):**")
    safe_image_display("images/dinara_confusion_matrix.png", "Confusion Matrix")