import streamlit as st
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
from pytorch_lightning import LightningModule
import albumentations as A
from albumentations.pytorch import ToTensorV2
import zipfile
import io
import shutil
import cv2

# Увеличиваем лимит размера загружаемых файлов до 500MB
from streamlit.runtime.uploaded_file_manager import UploadedFile
UploadedFile._max_size = 500 * 1024 * 1024  # 500MB

# Инициализация Streamlit
st.set_page_config(page_title="UNet Segmentation", layout="wide")

# Проверяем доступность GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
st.info(f"Используется устройство: {DEVICE.upper()}")

### MODEL ARCHITECTURE ###
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64,128,256,512)):
        super().__init__()
        self.downs = nn.ModuleList()
        for f in features:
            self.downs.append(nn.Sequential(
                nn.Conv2d(in_channels, f, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(f, f, 3, padding=1),
                nn.ReLU(inplace=True),
            ))
            in_channels = f
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]*2, features[-1]*2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.ups = nn.ModuleList()
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, stride=2))
            self.ups.append(nn.Sequential(
                nn.Conv2d(f*2, f, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(f, f, 3, padding=1),
                nn.ReLU(inplace=True),
            ))
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[-(idx//2 + 1)]
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx+1](x)
        return self.final_conv(x)

class UNetLitModule(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=1, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(in_channels, out_channels)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_acc = Accuracy(task="binary")
        self.val_acc   = Accuracy(task="binary")

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        imgs, masks = batch
        logits = self(imgs)
        loss   = self.loss_fn(logits, masks)
        preds  = (torch.sigmoid(logits) > 0.5).long()
        acc    = self.train_acc(preds, masks.long()) if stage=="train" else self.val_acc(preds, masks.long())
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc,   prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

### УЛУЧШЕННАЯ ЗАГРУЗКА МОДЕЛИ ###
@st.cache_resource
def load_model_from_path(model_path):
    """Загружает модель из указанного пути"""
    try:
        if not os.path.exists(model_path):
            st.error(f"Файл модели не найден: {model_path}")
            return None
            
        if os.path.getsize(model_path) == 0:
            st.error("Файл модели пустой. Удалите его и попробуйте снова.")
            return None
        
        model = UNetLitModule.load_from_checkpoint(model_path, map_location=DEVICE)
        model.eval()
        model.freeze()
        model.to(DEVICE)
        return model
        
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {str(e)}")
        return None

def setup_model_directory():
    """Создает папку для моделей и возвращает путь"""
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    return MODEL_DIR

def extract_zip_model(zip_file):
    """Извлекает модель из zip-архива"""
    MODEL_DIR = setup_model_directory()
    temp_path = os.path.join(MODEL_DIR, "unet_9_epoch.ckpt")
    
    try:
        with zipfile.ZipFile(io.BytesIO(zip_file.getvalue())) as z:
            for filename in z.namelist():
                if filename.endswith('.ckpt'):
                    with z.open(filename) as f, open(temp_path, 'wb') as out:
                        shutil.copyfileobj(f, out)
                    return temp_path
        return None
    except Exception as e:
        st.error(f"Ошибка распаковки архива: {str(e)}")
        return None

def handle_model_upload(uploaded_file):
    """Обрабатывает загруженный файл модели"""
    MODEL_DIR = setup_model_directory()
    
    if uploaded_file.name.endswith('.zip'):
        model_path = extract_zip_model(uploaded_file)
    else:
        model_path = os.path.join(MODEL_DIR, uploaded_file.name)
        with open(model_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    return model_path

### ФУНКЦИИ ДЛЯ ПРЕДСКАЗАНИЯ ###
def get_val_transform(size=256):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

@torch.no_grad()
def process_prediction(img_path, model, transform, threshold=0.5):
    """Обработка одного изображения с масштабированием маски"""
    # Загрузка изображения
    img = np.array(Image.open(img_path).convert("RGB"))
    original_size = img.shape[:2]  # Сохраняем оригинальный размер
    
    # Применяем трансформации
    transformed = transform(image=img)
    inp = transformed["image"].unsqueeze(0).to(DEVICE)
    
    # Получаем предсказание
    logits = model(inp)
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    
    # Масштабируем маску к оригинальному размеру
    pred_mask = (probs > threshold).astype(np.float32)
    pred_mask = cv2.resize(pred_mask, (original_size[1], original_size[0]), 
                         interpolation=cv2.INTER_LINEAR)
    
    # Средняя уверенность
    avg_conf = probs.mean() * 100
    
    return img, pred_mask, avg_conf

def create_visualization(img, pred_mask, avg_conf, alpha=0.4):
    """Создание визуализации с улучшенным отображением"""
    # Нормализация изображения
    if img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    
    # Создаем overlay для маски
    overlay = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0  # Красный канал
    overlay[..., 3] = pred_mask * alpha  # Альфа-канал
    
    # Создаем фигуру
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Оригинальное изображение
    ax1.imshow(img)
    ax1.set_title("Исходное изображение")
    ax1.axis('off')
    
    # Наложение маски
    ax2.imshow(img)
    ax2.imshow(overlay, alpha=alpha)
    ax2.set_title(f"Сегментация (уверенность: {avg_conf:.1f}%)")
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

### ОСНОВНОЕ ПРИЛОЖЕНИЕ ###
def main():
    st.title("UNet модель для сегментации аэрофотоснимков лесов")
    
    # Инициализация состояния сессии
    if "model" not in st.session_state:
        st.session_state.model = None
        st.session_state.transform = get_val_transform()
    
    # Сайдбар для загрузки модели
    st.sidebar.header("Загрузка модели")
    model_file = st.sidebar.file_uploader(
        "Выберите файл модели (unet_9_epoch.ckpt или .zip)", 
        type=["ckpt", "zip"],
        help="Модель можно скачать по ссылке: https://drive.google.com/uc?export=download&id=1zVFsP3idy0gk7JHfpnS52jdlhknCboXC"
    )
    
    # Обработка загруженной модели
    if model_file is not None:
        with st.spinner("Обработка модели..."):
            model_path = handle_model_upload(model_file)
            if model_path:
                st.session_state.model = load_model_from_path(model_path)
                if st.session_state.model:
                    file_size = os.path.getsize(model_path) / (1024 * 1024)
                    st.sidebar.success(f"Модель успешно загружена! Размер: {file_size:.1f}MB")
                else:
                    st.sidebar.error("Ошибка загрузки модели")
    
    if st.session_state.model is None:
        st.warning("Пожалуйста, загрузите модель для сегментации")
        return
    
    # Параметры сегментации
    st.sidebar.header("Параметры сегментации")
    threshold = st.sidebar.slider("Порог уверенности", 0.0, 1.0, 0.5, 0.01)
    alpha = st.sidebar.slider("Прозрачность маски", 0.0, 1.0, 0.4, 0.05)
    
    # Загрузка изображения
    st.header("Загрузите изображение для сегментации")
    uploaded_img = st.file_uploader(
        "Выберите изображение (JPG, PNG)", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_img is not None:
        try:
            with st.spinner("Обработка изображения..."):
                # Временный файл
                temp_path = "temp_input.png"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_img.getbuffer())
                
                # Обработка и визуализация
                img, pred_mask, avg_conf = process_prediction(
                    temp_path,
                    st.session_state.model,
                    st.session_state.transform,
                    threshold
                )
                
                fig = create_visualization(img, pred_mask, avg_conf, alpha)
                st.pyplot(fig)
                plt.close(fig)
                
                # Расчет площади
                area_pixels = pred_mask.sum()
                area_percent = area_pixels / (img.shape[0] * img.shape[1])
                st.success(f"Обнаружена площадь: {area_pixels:.0f} пикселей ({area_percent:.1%} изображения)")
                
                os.remove(temp_path)
                
        except Exception as e:
            st.error(f"Ошибка обработки: {str(e)}")
            st.error("Попробуйте загрузить другое изображение")

if __name__ == "__main__":
    main()