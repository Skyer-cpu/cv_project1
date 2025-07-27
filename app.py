import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'  # Решает проблемы с OpenCV
os.environ['DISPLAY'] = ':0'  # Эмулирует display для headless-режима

try:
    import cv2
    from ultralytics import YOLO
except ImportError as e:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.9.0.80", "ultralytics==8.1.0"], check=True)
    import cv2
    from ultralytics import YOLO
    
import streamlit as st
from PIL import Image
import pandas as pd
from ultralytics import YOLO
from PIL import Image, ImageFilter
import numpy as np
import os
import matplotlib.pyplot as plt
import requests
from io import BytesIO
try:
    import pytorch_lightning as pl
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch_lightning==2.1.0"])
    import pytorch_lightning as pl
# Настройки страницы
st.set_page_config(
    page_title="Команда ДДС",
    page_icon="🚀",
    layout="wide"
)

# Адаптивные стили CSS для светлой и тёмной тем
st.markdown("""
<style>
    :root {
        --primary-color: #3498db;
        --secondary-color: #7f8c8d;
        --text-color: var(--text-color);
        --bg-color: var(--background-color);
        --card-bg-light: #f5f7fa;
        --card-bg-dark: #2a2b2e;
        --quote-bg-light: #f8f9fa;
        --quote-bg-dark: #333438;
        --border-color: var(--border-color);
    }
    
    .header {
        font-size: 2.5em;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 30px;
        font-weight: 700;
    }
    .team-card {
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 30px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background-color: var(--card-bg);
        transition: all 0.3s ease;
    }
    @media (prefers-color-scheme: dark) {
        .team-card {
            background-color: var(--card-bg-dark);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
    }
    @media (prefers-color-scheme: light) {
        .team-card {
            background-color: var(--card-bg-light);
        }
    }
    .member-name {
        font-size: 1.8em;
        color: var(--primary-color);
        margin-bottom: 15px;
        font-weight: 600;
    }
    .member-role {
        font-size: 1.2em;
        color: var(--secondary-color);
        margin-bottom: 20px;
    }
    .quote {
        font-style: italic;
        padding: 15px;
        background-color: var(--quote-bg);
        border-left: 4px solid var(--primary-color);
        margin-top: 15px;
        border-radius: 0 8px 8px 0;
    }
    @media (prefers-color-scheme: dark) {
        .quote {
            background-color: var(--quote-bg-dark);
        }
    }
    @media (prefers-color-scheme: light) {
        .quote {
            background-color: var(--quote-bg-light);
            color: #333;
        }
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: var(--secondary-color);
        font-size: 0.9em;
        padding: 20px;
        border-top: 1px solid var(--border-color);
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<div class="header">Добро пожаловать в аналитический стримлит команды ДДС</div>', unsafe_allow_html=True)

# Карточка Сергея
with st.container():
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    st.markdown('<div class="member-name">Сергей</div>', unsafe_allow_html=True)
    st.markdown('<div class="member-role">Распознавание лиц</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        _ = st.image("https://via.placeholder.com/200", width=200)  # Исправлено здесь
    with col2:
        st.write("Сергей видит вас слишком хорошо, поэтому накладывает блюр — исключительно из эстетических соображений.")
        st.markdown('<div class="quote">"Ваше лицо настолько прекрасно, что мы решили сделать его... загадочным"</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Карточка Динары
with st.container():
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    st.markdown('<div class="member-name">Динара</div>', unsafe_allow_html=True)
    st.markdown('<div class="member-role">Нейроаналитика</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        _ = st.image("https://via.placeholder.com/200", width=200)  # Исправлено здесь
    with col2:
        st.write("Используя передовые алгоритмы глубокого обучения, мы проводим точнейший анализ медицинских изображений.")
        st.write("Каждый снимок обрабатывается с клинической точностью, потому что когда дело касается здоровья — важна каждая деталь.")
    st.markdown('</div>', unsafe_allow_html=True)

# Карточка Дмитрия
with st.container():
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    st.markdown('<div class="member-name">Дмитрий</div>', unsafe_allow_html=True)
    st.markdown('<div class="member-role">Аэрокосмические снимки</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        _ = st.image("https://via.placeholder.com/200", width=200)  # Исправлено здесь
    with col2:
        st.write("Дмитрий видит Землю с высоты, но до сих пор не нашёл ваш потерянный носок.")
        st.markdown('<div class="quote">"Да, этот пиксель — ваш дом. Нет, увеличить нельзя"</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Подвал
st.markdown("""
<div class="footer">
    Команда ДДС — мы как Илон Маск, только без ракет, нейрочипов и денег. Зато с душой!<br>
    © 2025 | Все права защищены, возможно, мы не уверены.
</div>
""", unsafe_allow_html=True)


# Разделитель перед "игрушками"
st.markdown("---")

# Заголовок для интерактивных элементов
st.caption("Эти инструменты ничего не делают, но если вам станет скучно, вы можете с ними поиграть 😊")

# Создаём 3 колонки для разных типов элементов
col1, col2, col3 = st.columns(3)

with col1:
    # Ползунок для "регулировки креативности"
    st.slider(
        "Уровень креативности", 
        min_value=0, 
        max_value=100, 
        value=42,
        key="creativity_slider",
        help="Этот ползунок ничего не регулирует, но выглядит важным"
    )

with col2:
    # Чекбоксы для "включения функций"
    st.checkbox(
        "Включить турбо-режим", 
        value=False,
        key="turbo_mode",
        help="Гарантированно ничего не ускоряет"
    )
    st.checkbox(
        "Активировать магию", 
        value=True,
        key="magic_mode",
        help="Магия включена по умолчанию (но всё равно не работает)"
    )

with col3:
    # Радиокнопки для "выбора стратегии"
    st.radio(
        "Стратегия анализа",
        options=["Мягкая", "Жёсткая", "Неопределённая"],
        index=2,
        key="strategy_radio",
        help="Выбор стратегии не влияет ни на что, кроме вашего настроения"
    )

# Добавляем кнопку с забавным эффектом
if st.button("✨ Нажми меня, если осмелишься", key="do_nothing_button"):
    st.balloons()  # Хотя бы что-то произойдёт!
    st.toast("Вы только что активировали ничего! Поздравляем!", icon="🎉")

# Секретная секция (появляется только если нажать кнопку)
if st.session_state.get("do_nothing_button", False):
    st.success("Вы обнаружили секретную функцию, которая тоже ничего не делает!")
    st.progress(0, text="Загрузка ничего...")