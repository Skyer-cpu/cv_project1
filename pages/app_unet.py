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

### УЛУЧШЕННАЯ ЗАГРУЗКА МОДЕЛИ ###
@st.cache_resource
def download_model():
    """Загружает модель с Google Drive"""
    MODEL_ID = "1zVFsP3idy0gk7JHfpnS52jdlhknCboXC"
    MODEL_DIR = "models"
    MODEL_NAME = "unet_9_epoch.ckpt"
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        try:
            with st.spinner('Скачивание модели с Google Drive (372 МБ)...'):
                # Альтернативный способ скачивания больших файлов
                output = f"{MODEL_DIR}/{MODEL_NAME}"
                gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", output, quiet=False)
                
                # Проверка успешности скачивания
                if not os.path.exists(MODEL_PATH):
                    st.error("Файл модели не был скачан. Попробуйте вручную:")
                    st.markdown(f"[Скачать модель](https://drive.google.com/uc?export=download&id={MODEL_ID})")
                    st.stop()
                    
                st.success("Модель успешно загружена!")
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {e}")
            st.error("Попробуйте вручную скачать модель и поместить в папку models/")
            st.stop()
    
    return MODEL_PATH

@st.cache_resource
def load_model():
    """Загружает и кэширует модель"""
    try:
        model_path = download_model()
        
        # Проверка файла модели
        if not os.path.exists(model_path):
            st.error(f"Файл модели не найден: {model_path}")
            st.stop()
            
        if os.path.getsize(model_path) == 0:
            st.error("Файл модели пустой. Удалите его и попробуйте снова.")
            st.stop()
        
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
        st.error(f"Критическая ошибка при загрузке модели: {str(e)}")
        st.stop()

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
    
    # Загрузка модели
    if "model" not in st.session_state:
        with st.spinner("Загрузка модели..."):
            st.session_state.model = load_model()
            st.session_state.transform = get_val_transform()
    
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
                    # Заменено use_column_width на use_container_width
                    st.image(img, caption=metric.split(".")[0], use_container_width=True)
                except:
                    st.warning(f"Не удалось загрузить {metric}")

if __name__ == "__main__":
    main()

