import streamlit as st
import os
import numpy as np
from PIL import Image
import requests
from io import BytesIO
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

### MODEL DOWNLOAD ###
@st.cache_resource
def download_model():
    """Загружает модель с Google Drive"""
    MODEL_URL = "https://drive.google.com/uc?id=1zVFsP3idy0gk7JHfpnS52jdlhknCboXC"
    MODEL_DIR = "models"
    MODEL_NAME = "unet_9_epoch.ckpt"
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        try:
            with st.spinner('Скачивание модели с Google Drive (372 МБ)...'):
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
                st.success("Модель успешно загружена!")
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {e}")
            st.stop()
    
    return MODEL_PATH

### MODEL LOADING ###
@st.cache_resource
def load_model():
    """Загружает и кэширует модель"""
    try:
        model_path = download_model()
        
        # Проверяем, что файл модели существует и не пустой
        if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
            st.error("Файл модели не найден или пустой. Пожалуйста, проверьте путь.")
            st.stop()
        
        # Загружаем модель с обработкой ошибок
        lit_model = UNetLitModule.load_from_checkpoint(
            model_path,
            map_location=DEVICE
        )
        
        lit_model.eval()
        lit_model.freeze()
        lit_model.to(DEVICE)
        
        st.success("Модель успешно загружена и готова к использованию!")
        return lit_model
        
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {str(e)}")
        st.stop()

### PREDICTION UTILS ###
def get_val_transform(size: int = 256):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

@torch.no_grad()
def overlay_prediction_only(img_path: str, model: torch.nn.Module, transform=None, 
                          threshold: float = 0.5, alpha: float = 0.4,
                          figsize: tuple = (12, 12), dpi: int = 100):
    try:
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]
        
        if transform is None:
            transform = get_val_transform()
            
        inp = transform(image=img)["image"].unsqueeze(0).to(DEVICE)
        
        logits = model(inp)
        probs  = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_mask = (probs > threshold).astype(float)
        avg_conf = probs.mean() * 100
        
        pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize((W, H), resample=Image.NEAREST)
        pred_mask = np.array(pred_mask_img) / 255.0
        
        base_img = img / 255.0
        overlay = np.zeros((H, W, 4))
        overlay[..., 2] = pred_mask  # Красный цвет для маски
        overlay[..., 3] = alpha * pred_mask
        
        fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        ax[0].imshow(base_img)
        ax[0].axis("off")
        ax[0].set_title("Исходное изображение")
        
        ax[1].imshow(base_img)
        ax[1].imshow(overlay)
        ax[1].axis("off")
        ax[1].set_title(f"Сегментация (уверенность: {avg_conf:.1f}%)")
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Ошибка при обработке изображения: {str(e)}")
        return None

### STREAMLIT APP ###
def main():
    st.set_page_config(page_title="UNet Segmentation", layout="wide")
    st.title("UNet модель для сегментации аэрофотоснимков лесов")
    
    # Загрузка модели
    if "model" not in st.session_state:
        st.session_state.model = load_model()
    
    # Настройки в сайдбаре
    st.sidebar.header("Параметры сегментации")
    threshold = st.sidebar.slider("Порог уверенности", 0.0, 1.0, 0.5, 0.01)
    alpha = st.sidebar.slider("Прозрачность маски", 0.0, 1.0, 0.4, 0.05)
    
    # Загрузка изображения
    st.header("Загрузите изображение для сегментации")
    uploaded = st.file_uploader("Выберите файл", type=["jpg", "png", "jpeg"], accept_multiple_files=False)
    
    if uploaded is not None:
        try:
            # Сохраняем временный файл
            tmp_path = "temp_input.png"
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            
            # Обработка изображения
            with st.spinner("Обработка изображения..."):
                fig = overlay_prediction_only(
                    img_path=tmp_path,
                    model=st.session_state.model,
                    transform=get_val_transform(),
                    threshold=threshold,
                    alpha=alpha,
                    figsize=(10, 5),
                    dpi=100
                )
                
                if fig is not None:
                    st.pyplot(fig)
                    plt.close(fig)
            
            # Удаляем временный файл
            os.remove(tmp_path)
            
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {str(e)}")
    
    # Отображение метрик обучения
    st.header("Метрики обучения модели")
    metrics_dir = "images/unet"
    
    if os.path.isdir(metrics_dir):
        cols = st.columns(2)
        metric_images = sorted([f for f in os.listdir(metrics_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        
        for idx, img_name in enumerate(metric_images):
            img_path = os.path.join(metrics_dir, img_name)
            with cols[idx % 2]:
                try:
                    st.image(
                        Image.open(img_path),
                        caption=os.path.splitext(img_name)[0].replace("_", " ").title(),
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Не удалось загрузить изображение {img_name}: {str(e)}")
    else:
        st.warning("Директория с метриками обучения не найдена.")

if __name__ == "__main__":
    main()