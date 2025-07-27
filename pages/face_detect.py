import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageFilter
import numpy as np
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# --- Конфигурация ---
MODEL_DIR = "models"
MODEL_NAME = "best_face.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# --- Заголовок и описание приложения Streamlit ---
st.set_page_config(page_title="Приложение для размытия лиц", layout="wide")
st.title("Обнаружение и размытие лиц с помощью YOLOv8(nano)")
st.write("Загрузите изображение для обнаружения и размытия лиц с использованием обученной модели YOLOv8.")

# --- Загрузка обученной модели ---
@st.cache_resource # Кэшировать модель, чтобы избежать перезагрузки при каждом запуске
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Ошибка: Файл модели не найден по пути {path}. Убедитесь, что '{MODEL_NAME}' находится в каталоге '{MODEL_DIR}'.")
        return None
    try:
        model = YOLO(path)
        st.success(f"Модель '{MODEL_NAME}' успешно загружена! Можно приступать к использованию.")
        return model
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        return None

trained_model = load_model(MODEL_PATH)

# --- Функция размытия лиц ---
def blur_faces(image, model, blur_radius=20):
    if model is None:
        st.error("Модель не загружена. Невозможно размыть лица.")
        return image

    img_pil = image.convert('RGB')
    img_np = np.array(img_pil)

    try:
        results = model(img_np)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Убедиться, что координаты находятся в пределах изображения
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_pil.width, x2)
                y2 = min(img_pil.height, y2)

                face_region = img_pil.crop((x1, y1, x2, y2))
                blurred_face = face_region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                img_pil.paste(blurred_face, (x1, y1))
        return img_pil
    except Exception as e:
        st.error(f"Ошибка во время обнаружения или размытия лиц: {e}")
        return image

# --- Загрузка и обработка изображения ---
st.header("Загрузите изображение для размытия лиц")
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Исходное изображение", use_column_width=True)

    if trained_model:
        st.write("Обработка изображения...")
        blurred_image = blur_faces(input_image, trained_model)
        st.image(blurred_image, caption="Размытое изображение", use_column_width=True)
        st.download_button(
            label="Скачать размытое изображение",
            data=open("temp_blurred_image.png", "rb").read() if os.path.exists("temp_blurred_image.png") else b'',
            file_name="blurred_image.png",
            mime="image/png"
        )
        # Временно сохранить размытое изображение для скачивания
        blurred_image.save("temp_blurred_image.png")
    else:
        st.warning("Модель не загружена. Проверьте путь к модели и попробуйте снова.")

# --- Раздел с результатами обучения ---
st.header("Результаты и метрики обучения")
st.write("Ниже представлены графики, сгенерированные во время обучения модели обнаружения лиц YOLOv8.")

def show_f1_analysis():
    st.header("Анализ F1-Confidence Curve")
    
    # Загрузка изображения
    f1_curve = Image.open('images/BoxF1_curveSP.png')
    st.image(f1_curve, caption='Кривая F1-Confidence', use_container_width=True)
    
    # Анализ
    st.markdown("""
    ### Основные наблюдения:
    - **Пик F1=0.85** достигается при **confidence=0.33** - оптимальный баланс между точностью и полнотой
    - **При confidence > 0.6** производительность резко падает - модель становится излишне строгой
    
    ### Рекомендации по настройке порога:
    """)
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("Оптимальный диапазон", "0.3-0.4")
    with cols[1]:
        st.metric("Для точности", "~0.5")
    with cols[2]:
        st.metric("Для полноты", "~0.2")
    
    st.markdown("""
    ### Ключевой вывод:
    Кривая имеет асимметричный характер - после пика производительность снижается быстрее при увеличении confidence, 
    чем при его уменьшении. Это указывает на склонность модели к излишней осторожности: при высоких порогах 
    она теряет больше истинных обнаружений, чем исключает ложные.
    """)

# Вызов функции
show_f1_analysis()



def show_precision_analysis():
    st.header("Анализ Precision-Confidence Curve")
    
    # Загрузка изображения
    precision_curve = Image.open('images/BoxP_curveSP.png')
    st.image(precision_curve, caption='Кривая Precision-Confidence', use_container_width=True)
    
    # Анализ
    st.markdown("""
    ### Основные наблюдения:
    - **Максимальная точность 1.00** достигается при **confidence=0.93** - идеальная точность предсказаний
    - **При confidence < 0.7** точность постепенно снижается - появляется больше ложных срабатываний
    
    ### Интерпретация кривой:
    """)
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("Максимальная точность", "1.00")
    with cols[1]:
        st.metric("Достигается при", "confidence=0.93")
    with cols[2]:
        st.metric("Высокая точность (>0.9)", "при confidence>0.8")
    
    st.markdown("""
    ### Ключевой вывод:
    Кривая показывает, что модель достигает идеальной точности при очень высоких порогах уверенности.
    Однако такие строгие пороги могут значительно снизить полноту (recall). Для баланса между точностью
    и полнотой рекомендуется выбирать confidence в диапазоне 0.6-0.8.
    """)

# Вызов функции
show_precision_analysis()


def show_recall_analysis():
    st.header("Анализ Recall-Confidence Curve")
    
    # Загрузка изображения
    recall_curve = Image.open('images/BoxR_curveSP.png')
    st.image(recall_curve, caption='Кривая Recall-Confidence', use_container_width=True)
    
    # Анализ
    st.markdown("""
    ### Основные наблюдения:
    - **Максимальный recall 0.93** достигается при **confidence=0.000** - модель обнаруживает почти все объекты без фильтрации
    - **При confidence > 0.5** recall резко падает - модель становится слишком консервативной
    
    ### Ключевые показатели:
    """)
    
    cols = st.columns(2)
    with cols[0]:
        st.metric("Пиковый recall", "0.93")
    with cols[1]:
        st.metric("При confidence", "0.000")
    
    st.markdown("""
    ### Практические рекомендации:
    - Для максимального охвата (скрининг): **confidence < 0.2**
    - Для баланса точности/полноты: **0.2-0.4**
    - Для высокоточной работы: **> 0.6** (но recall будет низким)
    
    ### Вывод:
    Кривая демонстрирует классический компромисс между recall и confidence - чем строже порог уверенности,
    тем больше объектов пропускает модель. Оптимальный порог следует выбирать исходя из задачи:
    обнаружение всех возможных объектов vs минимизация ложных срабатываний.
    """)

# Вызов функции
show_recall_analysis()






def show_pr_curve_analysis():
    st.header("Анализ кривой Precision-Recall для системы распознавания лиц")
    
    image_path = 'images/BoxPR_curveSP.png'
    try:
        pr_curve_image = Image.open(image_path)
        st.image(pr_curve_image, caption='Кривая Precision-Recall для распознавания лиц', use_container_width=True)
    except FileNotFoundError:
        st.error(f"Ошибка: Файл изображения '{image_path}' не найден. Пожалуйста, убедитесь, что он находится в указанной директории.")
        return
    
    # Анализ
    st.markdown("""
    ### Основные наблюдения для системы распознавания лиц:
    Кривая Precision-Recall показывает эффективность модели в задаче обнаружения лиц:

    - **Высокая точность (Precision ~1.0)** при Recall до ~0.8 означает, что система почти не дает ложных срабатываний при обнаружении большинства лиц
    - **Резкое падение точности** после Recall=0.85 указывает на сложные случаи (размытые, частично закрытые или плохо освещенные лица)
    - **mAP@0.5=0.879** свидетельствует о высокой надежности системы в стандартных условиях
    """)
    
    cols = st.columns(2)
    with cols[0]:
        st.metric("Средняя точность (mAP@0.5)", "0.879")
        st.caption("Оценка качества для всех типов лиц")
    with cols[1]:
        st.metric("Производительность для лиц", "0.879")
        st.caption("Включая сложные случаи (маски, плохое освещение)")
    
    st.markdown("""
    ### Рекомендации по настройке для распознавания лиц:

    #### 1. Для систем контроля доступа (приоритет точности):
    - **Диапазон Recall: 0.7-0.8**
    - Минимизация ложных пропусков посторонних
    - Примеры: банковские системы, secure-зоны

    #### 2. Для видеонаблюдения (баланс точности/полноты):
    - **Диапазон Recall: 0.8-0.85**
    - Компромисс между обнаружением и ложными срабатываниями
    - Примеры: розыск людей, аналитика посетителей

    #### 3. Для массовых мероприятий (приоритет полноты):
    - **Диапазон Recall: 0.85+**
    - Максимальное покрытие с дополнительной ручной проверкой
    - Примеры: поиск пропавших, идентификация в толпе

    ### Оптимизация для разных сценариев:
    - **Хорошее освещение:** можно использовать более строгие пороги
    - **Сложные условия:** снижать порог уверенности
    - **Работа с масками:** требует отдельной калибровки модели
    """)

# Вызов функции
show_pr_curve_analysis()


def show_confusion_matrix_simple():
    st.header("Анализ матрицы ошибок для распознавания лиц")
    
    try:
        confusion_matrix_img = Image.open('images/confusion_matrixSP.png')
        st.image(confusion_matrix_img, use_container_width=True)
    except FileNotFoundError:
        st.error("Файл изображения не найден")
        return
    
    # Анализ остается таким же
    st.markdown("""
    ### Ключевые показатели:
    - Верно распознанные лица: 8604
    - Ложные срабатывания: 1695
    - Пропущенные лица: 1398
    """)

show_confusion_matrix_simple()


import streamlit as st
from PIL import Image

def show_normalized_confusion_matrix():
    st.header("Анализ нормализованной матрицы ошибок")
    
    # Загрузка и отображение только вашего графика
    try:
        conf_matrix_img = Image.open('images/confusion_matrix_normalizedSP.png')
        st.image(conf_matrix_img, 
                caption='Нормализованная матрица ошибок распознавания лиц', 
                use_container_width=True)
    except FileNotFoundError:
        st.error("Файл не найден. Проверьте путь: 'images/confusion_matrix_normalized.png'")
        return
    
    # Анализ данных из графика
    st.markdown("""
    ### Ключевые показатели:
    - **Точность распознавания лиц (facc):** 84%
    - **Ошибки фона (background):** 16% 
    - **Ложные обнаружения лиц (face):** 20%
    - **Пропущенные лица:** 40%

    ### Интерпретация:
    1. **Сильные стороны:**
       - Высокая точность (84%) правильного распознавания лиц
       - Умеренное количество ложных срабатываний (20%)

    2. **Проблемные зоны:**
       - Довольно высокий процент пропущенных лиц (40%)
       - 16% случаев, когда фон ошибочно принимается за лицо

    ### Рекомендации:
    - Для систем безопасности: повысить порог уверенности (снизит ложные срабатывания)
    - Для видеонаблюдения: добавить аугментации с масками/размытиями (уменьшит пропуски)
    - Оптимальный баланс: порог confidence ~0.4-0.5
    """)

show_normalized_confusion_matrix()


def plot_map_metrics():
    st.header("Сравнение mAP50 и mAP50-95 по эпохам")
    
    # Данные из вашего обучения
    epochs = list(range(1, 21))
    map50 = [
        0.775, 0.793, 0.798, 0.816, 0.828, 0.838, 0.844, 0.848, 0.853, 0.857,
        0.857, 0.863, 0.862, 0.867, 0.868, 0.871, 0.874, 0.875, 0.877, 0.879
    ]
    map50_95 = [
        0.464, 0.482, 0.504, 0.517, 0.525, 0.537, 0.551, 0.549, 0.558, 0.560,
        0.561, 0.564, 0.564, 0.568, 0.575, 0.578, 0.582, 0.580, 0.585, 0.586
    ]

    # Создаем DataFrame
    data = pd.DataFrame({
        'Epoch': epochs,
        'mAP50': map50,
        'mAP50-95': map50_95
    })

    # Строим график
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Линия mAP50
    ax.plot(data['Epoch'], data['mAP50'], 
            label='mAP50 (IoU=0.5)', 
            marker='o', 
            color='blue',
            linewidth=2)
    
    # Линия mAP50-95
    ax.plot(data['Epoch'], data['mAP50-95'], 
            label='mAP50-95 (среднее для IoU 0.5:0.95)', 
            marker='s', 
            color='green',
            linewidth=2)
    
    # Настройки графика
    ax.set_title('Динамика метрик mAP в процессе обучения', fontsize=14, pad=20)
    ax.set_xlabel('Номер эпохи', fontsize=12)
    ax.set_ylabel('Значение метрики', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    ax.set_xticks(epochs)
    ax.set_ylim(0.4, 0.9)
    
    # Выделяем максимальные значения
    max_map50 = data['mAP50'].max()
    max_map50_95 = data['mAP50-95'].max()
    
    ax.axhline(y=max_map50, color='blue', linestyle=':', alpha=0.5)
    ax.axhline(y=max_map50_95, color='green', linestyle=':', alpha=0.5)
    
    # Аннотации
    ax.annotate(f'Max mAP50: {max_map50:.3f}', 
                xy=(20, max_map50), 
                xytext=(15, 0.75),
                color='blue',
                arrowprops=dict(facecolor='blue', shrink=0.05))
    
    ax.annotate(f'Max mAP50-95: {max_map50_95:.3f}', 
                xy=(20, max_map50_95), 
                xytext=(15, 0.5),
                color='green',
                arrowprops=dict(facecolor='green', shrink=0.05))
    
    st.pyplot(fig)
    
    # Анализ
    st.markdown("""
    ### Ключевые различия метрик:
    - **mAP50**:
      - фиксированный IoU=0.5
      - максимальное значение: **0.879**
    
    - **mAP50-95**:
      - Усреднение по IoU от 0.5 до 0.95 с шагом 0.05
      - максимальное значение: **0.586**
    
    ### Интерпретация результатов:
    1. **Общий рост обеих метрик** показывает эффективное обучение
    2. **Разрыв между метриками (~0.3)** типичен для задач детекции
    3. **После 15 эпохи** рост замедляется - возможен выход на плато, остановка обучения
    
    ### Рекомендации:
    - Для задач, где важна **общая детекция** (а не точные bbox) - ориентация на mAP50
    - Для задач, где критична **точность позиционирования** - ориентация на mAP50-95
    - Компромисс обучения - **эпохи 15-20** (оба показателя близки к максимуму)
    """)

plot_map_metrics()


st.markdown("---")
st.write("Разработано с ❤️ использованием Streamlit и Ultralytics YOLOv8(nano).")

