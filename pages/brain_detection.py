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

# Устанавливаем настройки страницы Streamlit
st.set_page_config(page_title="Детекция опухолей мозга", layout="wide")
st.title("🧠 Детекция опухолей мозга (YOLOv8)") # Заголовок остается YOLOv8, так как это общепринятая актуальная версия

# --- Конфигурация модели ---
MODEL_DIR = "models"
# Оставляем имя файла модели как указано, но помним, что проблема не в названии.
MODEL_NAME = "dinara_yolov11_best.pt" 
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# --- Загрузка обученной модели (с кэшированием) ---
@st.cache_resource
def load_model(path):
    """
    Загружает модель YOLO с указанного пути.
    Кэширует модель, чтобы избежать повторной загрузки.
    """
    if not os.path.exists(path):
        st.error(f"Ошибка: Файл модели не найден по пути **{path}**. "
                 f"Убедитесь, что '{MODEL_NAME}' находится в каталоге '{MODEL_DIR}'.")
        return None
    try:
        # Проверяем доступность GPU и выбираем устройство
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.info(f"Используется устройство: **{device.upper()}**")
        
        # Загружаем модель YOLO
        model = YOLO(path)
        model.to(device) # Перемещаем модель на выбранное устройство
        
        # Проверяем, что модель загрузилась
        if model is not None:
            st.success(f"Модель '{MODEL_NAME}' успешно загружена! Можно приступать к использованию.")
            st.info(f"**Важно:** Если вы столкнулись с ошибкой 'Can't get attribute ...', "
                    f"это может быть связано с несовместимостью версий библиотеки `ultralytics`. "
                    f"Пожалуйста, **установите версию `ultralytics`, соответствующую той, на которой обучалась модель.**")
            return model
        else:
            st.error("Не удалось загрузить модель. Возможно, файл модели поврежден или не является допустимой моделью YOLO.")
            return None
    except Exception as e:
        # Более информативное сообщение об ошибке
        st.error(f"**Критическая ошибка при загрузке модели:** {str(e)}. "
                 f"Эта ошибка ('Can't get attribute...') указывает на **несовместимость версий `ultralytics`**."
                 f"Убедитесь, что версия `ultralytics` в вашем окружении (используйте `pip show ultralytics`) "
                 f"совпадает с версией, использованной для обучения модели.")
        return None

# Загружаем модель при запуске приложения
model = load_model(MODEL_PATH)

# ===== Секция загрузки изображений =====
st.header("Загрузите изображение для детекции")
tab1, tab2 = st.tabs(["📂 Загрузить файл", "🌐 По ссылке"])

uploaded_files = []
image_url = ""

with tab1:
    uploaded_files = st.file_uploader(
        "Загрузите одно или несколько изображений (JPG, PNG):",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

with tab2:
    image_url = st.text_input("Или вставьте прямую ссылку на изображение (JPG/PNG):")

# Обработка загруженных файлов и URL
imgs = []
if uploaded_files:
    for file in uploaded_files:
        try:
            img = Image.open(file)
            imgs.append(img)
        except Exception as e:
            st.error(f"Ошибка при открытии файла **{file.name}**: {str(e)}")

if image_url:
    try:
        # Добавляем user-agent для имитации запроса из браузера
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(image_url, timeout=10, headers=headers) # Увеличиваем таймаут
        response.raise_for_status() # Вызывает исключение для плохих статусов HTTP
        img = Image.open(BytesIO(response.content))
        imgs.append(img)
    except requests.exceptions.Timeout:
        st.error("Ошибка: Время ожидания загрузки изображения по ссылке истекло. Попробуйте другую ссылку.")
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка загрузки изображения по ссылке: {str(e)}. Убедитесь, что ссылка корректна и доступна.")
    except Exception as e:
        st.error(f"Ошибка обработки изображения по ссылке: {str(e)}")

# ===== Инференс и отображение результатов =====
if model is not None and imgs:
    st.markdown("---")
    st.header("Результаты детекции")
    for img in imgs:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption="Оригинальное изображение", use_container_width=True)
        
        try:
            # YOLO предсказание
            with st.spinner("Выполняется детекция..."):
                results = model.predict(img, conf=0.25) # conf - порог уверенности для детекции
            
            with col2:
                for r in results:
                    # Проверяем, есть ли аннотированное изображение
                    if hasattr(r, 'plot'):
                        # plot() возвращает numpy массив изображения с отрисованными боксами
                        annotated_img = r.plot()
                        st.image(annotated_img, caption="Результат детекции", use_container_width=True)
                    else:
                        st.warning("Модель не смогла создать аннотированное изображение.")
                    
                    # Показываем информацию о детекции
                    if len(r.boxes) > 0:
                        st.success(f"Обнаружено **{len(r.boxes)}** опухолей.")
                        # Опционально: можно вывести подробности по каждой опухоли
                        for i, box in enumerate(r.boxes):
                            conf = box.conf.item() * 100
                            cls = int(box.cls.item())
                            # Важно: model.names может быть None, если модель не имеет определенных имен классов
                            label = model.names[cls] if hasattr(model, 'names') and model.names else f"Класс {cls}" 
                            st.write(f"- Опухоль {i+1}: Класс **{label}**, Уверенность **{conf:.2f}%**")
                    else:
                        st.info("Опухоли не обнаружены на этом изображении.")
        except Exception as e:
            st.error(f"Ошибка при выполнении детекции для изображения: {str(e)}")

# ===== Информация о модели и метрики =====
st.markdown("---")
with st.expander("ℹ️ Информация о модели и метрики"):
    st.markdown("""
    ### **YOLOv8** использовалась для детекции опухолей на MR-снимках.

    - 📦 **Размер обучающей выборки:** 310 изображений 
    - 📦 **Размер валидационной выборки:** 75 изображений 
    - 🔁 **Эпохи обучения:** 50 эпох * 3 модели = 150 
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
                st.warning(f"Изображение не найдено: {image_path}. "
                           f"Убедитесь, что оно находится по пути '{image_path}'.")
        except Exception as e:
            st.error(f"Ошибка при загрузке изображения **{image_path}**: {str(e)}")

    # Отображение графиков
    st.markdown("#### **Графики метрик обучения:**")
    st.markdown("---")
    
    st.markdown("##### **ГРАФИК №1: mAP - детекция опухолей мозга (YOLOv8)**")
    safe_image_display("images/dinara_map.jpg", "График mAP")
    
    st.markdown("##### **ГРАФИК №2: box_f1_curve - детекция опухолей мозга (YOLOv8)**")
    safe_image_display("images/dinara_BoxF1_curve.png", "F1 Curve для боксов")
    
    st.markdown("##### **ГРАФИК №3: PR-кривая - детекция опухолей мозга (YOLOv8)**")
    safe_image_display("images/dinara_BoxPR_curve.png", "PR-кривая для боксов")
    
    st.markdown("##### **ГРАФИК №4: Матрица ошибок - детекция опухолей мозга (YOLOv8)**")
    safe_image_display("images/dinara_confusion_matrix.png", "Матрица ошибок")


