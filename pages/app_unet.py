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
import gdown

# Инициализация Streamlit (должен быть первым)
st.set_page_config(page_title="UNet Segmentation", layout="wide")

# Проверяем доступность GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

### MODEL ARCHITECTURE (без изменений) ###
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

### МОДИФИЦИРОВАННАЯ ЗАГРУЗКА МОДЕЛИ ###
@st.cache_resource
def load_model_from_path(model_path):
    """Загружает модель из указанного пути"""
    try:
        # Проверка файла модели
        if not os.path.exists(model_path):
            st.error(f"Файл модели не найден: {model_path}")
            return None
            
        if os.path.getsize(model_path) == 0:
            st.error("Файл модели пустой. Удалите его и попробуйте снова.")
            return None
        
        # Загрузка модели
        model = UNetLitModule.load_from_checkpoint(
            model_path,
            map_location=DEVICE
        )
        
        model.eval()
        model.freeze()
        model.to(DEVICE)
        
        return model
        
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {str(e)}")
        return None

def get_default_model():
    """Пытается загрузить модель по умолчанию"""
    MODEL_DIR = "models"
    MODEL_NAME = "unet_9_epoch.ckpt"
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if os.path.exists(MODEL_PATH):
        return load_model_from_path(MODEL_PATH)
    return None

### ФУНКЦИИ ДЛЯ ПРЕДСКАЗАНИЯ ###
def get_val_transform(size=256):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

@torch.no_grad()
def process_prediction(img_path, model, transform, threshold=0.5):
    """Обработка одного изображения"""
    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    
    # Преобразование изображения
    inp = transform(image=img)["image"].unsqueeze(0).to(DEVICE)
    
    # Предсказание
    logits = model(inp)
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    pred_mask = (probs > threshold).astype(float)
    avg_conf = probs.mean() * 100
    
    return img, pred_mask, avg_conf

def create_visualization(img, pred_mask, avg_conf, alpha=0.4):
    """Создание визуализации результатов"""
    H, W = img.shape[:2]
    base_img = img / 255.0
    
    # Создание маски с прозрачностью
    overlay = np.zeros((H, W, 4))
    overlay[..., 2] = pred_mask  # Красный канал
    overlay[..., 3] = alpha * pred_mask  # Альфа-канал
    
    # Создание фигуры
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Оригинальное изображение
    ax[0].imshow(base_img)
    ax[0].axis("off")
    ax[0].set_title("Исходное изображение")
    
    # Результат сегментации
    ax[1].imshow(base_img)
    ax[1].imshow(overlay)
    ax[1].axis("off")
    ax[1].set_title(f"Сегментация (уверенность: {avg_conf:.1f}%)")
    
    plt.tight_layout()
    return fig

### ОСНОВНОЕ ПРИЛОЖЕНИЕ ###
def main():
    st.title("UNet модель для сегментации аэрофотоснимков лесов")
    st.info(f"Используется устройство: {DEVICE.upper()}")
    
    # Добавляем возможность загрузки модели вручную
    st.sidebar.header("Загрузка модели")
    model_file = st.sidebar.file_uploader(
        "Загрузите модель (unet_9_epoch.ckpt)", 
        type=["ckpt", "pth"],
        help="Скачайте модель по ссылке: https://drive.google.com/uc?export=download&id=1zVFsP3idy0gk7JHfpnS52jdlhknCboXC"
    )
    
    # Инициализация модели
    if "model" not in st.session_state:
        # Сначала пробуем загрузить модель по умолчанию
        default_model = get_default_model()
        if default_model is not None:
            st.session_state.model = default_model
            st.session_state.transform = get_val_transform()
            st.sidebar.success("Используется модель по умолчанию")
        else:
            st.session_state.model = None
    
    # Если пользователь загрузил свою модель
    if model_file is not None:
        try:
            # Сохраняем временный файл модели
            MODEL_DIR = "models"
            os.makedirs(MODEL_DIR, exist_ok=True)
            temp_model_path = os.path.join(MODEL_DIR, "uploaded_model.ckpt")
            
            with open(temp_model_path, "wb") as f:
                f.write(model_file.getbuffer())
            
            # Загружаем модель
            with st.spinner("Загрузка модели..."):
                uploaded_model = load_model_from_path(temp_model_path)
                if uploaded_model is not None:
                    st.session_state.model = uploaded_model
                    st.session_state.transform = get_val_transform()
                    st.sidebar.success("Модель успешно загружена!")
        except Exception as e:
            st.sidebar.error(f"Ошибка загрузки модели: {str(e)}")
    
    # Проверяем, загружена ли модель
    if st.session_state.get("model") is None:
        st.warning("Модель не загружена. Пожалуйста, загрузите модель в разделе 'Загрузка модели'")
        st.markdown("""
            ### Инструкция по загрузке модели:
            1. Скачайте модель по [этой ссылке](https://drive.google.com/uc?export=download&id=1zVFsP3idy0gk7JHfpnS52jdlhknCboXC)
            2. Нажмите "Browse files" в разделе "Загрузка модели" слева
            3. Выберите скачанный файл `unet_9_epoch.ckpt`
            4. Дождитесь загрузки модели
        """)
        return
    
    # Параметры в сайдбаре
    st.sidebar.header("Параметры сегментации")
    threshold = st.sidebar.slider("Порог уверенности", 0.0, 1.0, 0.5, 0.01)
    alpha = st.sidebar.slider("Прозрачность маски", 0.0, 1.0, 0.4, 0.05)
    
    # Загрузка изображения
    st.header("Загрузите изображение для сегментации")
    uploaded = st.file_uploader("Выберите файл", type=["jpg", "png", "jpeg"])
    
    if uploaded is not None:
        try:
            # Сохраняем временный файл
            temp_path = "temp_input.png"
            with open(temp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            
            # Обработка изображения
            with st.spinner("Анализ изображения..."):
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
            tumor_area = pred_mask.sum()
            st.info(f"Обнаружена площадь: {tumor_area:.0f} пикселей ({tumor_area/(img.shape[0]*img.shape[1]):.1%} изображения)")
            
            # Удаление временного файла
            os.remove(temp_path)
            
        except Exception as e:
            st.error(f"Ошибка обработки: {str(e)}")
    
    # Метрики обучения
    st.header("Метрики обучения")
    metrics_dir = "images/unet"
    if os.path.exists(metrics_dir):
        cols = st.columns(2)
        metrics = sorted([f for f in os.listdir(metrics_dir) if f.endswith((".png", ".jpg", ".jpeg"))])
        
        for i, metric in enumerate(metrics):
            with cols[i % 2]:
                try:
                    img = Image.open(os.path.join(metrics_dir, metric))
                    st.image(img, caption=metric.split(".")[0], use_container_width=True)
                except:
                    st.warning(f"Не удалось загрузить {metric}")

if __name__ == "__main__":
    main()