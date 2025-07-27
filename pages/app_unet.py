### IMPORT ###
import streamlit as stimport
import os
import streamlit as st
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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

### Load checkpoint ###
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64,128,256,512)):
        super().__init__()
        # определение encoder
        self.downs = nn.ModuleList()
        for f in features:
            self.downs.append(nn.Sequential(
                nn.Conv2d(in_channels, f, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(f, f, 3, padding=1),
                nn.ReLU(inplace=True),
            ))
            in_channels = f
        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]*2, features[-1]*2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # decoder
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

# путь к чекпойнту
ckpt_path = "../models/unet_9_epoch.ckpt"

# воссоздаём архитектуру (должна совпадать с той, что использовалась при обучении)
base_model = UNet(in_channels=3, out_channels=1)  # 1 класс: лес/фон[1]

# восстанавливаем веса
lit_model: LightningModule = UNetLitModule.load_from_checkpoint(
    ckpt_path, 
    model=base_model,       # передаём базовую сеть
    lr=1e-3                 # остальные гиперпараметры
)

lit_model.eval()
lit_model.freeze()          # фиксируем параметры для чистого инференса
lit_model.to(DEVICE)

def get_val_transform(size: int = 256):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

### Predict results ###
@torch.no_grad()
def overlay_prediction_only(img_path: str,
                            model: torch.nn.Module,
                            transform=None,
                            threshold: float = 0.5,
                            alpha: float = 0.4,
                            figsize: tuple = (20, 20),
                            dpi: int = 150):
    """
    Отображает одно тестовое изображение с наложенной предсказанной моделью маской,
    когда истинная маска отсутствует.
    
    Параметры:
    - img_path: путь к RGB-изображению
    - model: LitModule или nn.Module, возвращающий логиты B×1×H×W
    - transform: функция/Albumentations.Transform, приводящая изображение к тензору C×H×W
    - threshold: порог бинаризации вероятностей
    - alpha: прозрачность слоя предсказанной маски
    - figsize: размер фигуры в дюймах
    - dpi: разрешение фигуры
    """
    # Загрузка и подготовка изображения
    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    inp = transform(image=img)["image"].unsqueeze(0).to(next(model.parameters()).device)  # B×C×h×w
    
    # Предсказание вероятностей и бинаризация
    logits = model(inp)                       # B×1×h×w
    probs  = torch.sigmoid(logits)[0, 0].cpu().numpy()  # h×w
    pred_mask = (probs > threshold).astype(float)      # h×w

    # Усреднённая уверенность по всем пикселям
    avg_conf = probs.mean() * 100  # в процентах
    
    # Приведение маски к размеру оригинала, если трансформ масштабировал
    pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize((W, H), resample=Image.NEAREST)
    pred_mask = np.array(pred_mask_img) / 255.0        # H×W
    
    # Подготовка фонового изображения
    base_img = img / 255.0
    
    # Создание RGBA-слоя для предсказанной маски (зелёный)
    overlay = np.zeros((H, W, 4))
    overlay[..., 2] = pred_mask     # зелёный канал
    overlay[..., 3] = alpha * pred_mask  # альфа-канал
    
    # Визуализация
    fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
    ax[0].imshow(base_img)
    ax[1].imshow(base_img)
    ax[1].imshow(overlay)
    ax[0].axis("off")
    ax[1].axis("off")
    ax[0].set_title("Исходное изображение")
    ax[1].set_title(f"Изображение с предсказанием (уверенность модели {avg_conf:.1f}%)")
    return fig

### Streamlit-приложение ###

@st.cache_resource
def load_model(ckpt_path: str):
    """Кэшируем модель, чтобы не перезагружать."""
    base = UNet(in_channels=3, out_channels=1)
    lit  = UNetLitModule.load_from_checkpoint(
        ckpt_path,
        in_channels=3, out_channels=1, lr=1e-3
    )
    lit.eval()
    lit.freeze()
    lit.to(DEVICE)

    return lit

def main():
    st.title("UNet модель обученная на сегментацию изображений аэрофотоснимков лесов")

    st.sidebar.header("Параметры")
    ckpt_path = st.sidebar.text_input("Путь к .ckpt", value="../models/unet_9_epoch.ckpt")
    if st.sidebar.button("Загрузить модель"):
        if os.path.isfile(ckpt_path):
            st.session_state.model = load_model(ckpt_path)
            st.success("Модель загружена")
        else:
            st.error("Чекпойнт не найден")

    threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5, 0.01)
    alpha     = st.sidebar.slider("Прозрачность", 0.0, 1.0, 0.4, 0.05)

    uploaded = st.file_uploader("Загрузите изображение", type=["jpg","png","jpeg"])
    if uploaded:
        # Конвертируем загруженный файл в PIL.Image и numpy-массив
        img_pil = Image.open(uploaded).convert("RGB")
        tmp_path = "tmp_input.png"
        img_pil.save(tmp_path)
        # Теперь можно показать входное изображение
        # st.image(img_pil, caption="Входное изображение", use_container_width=True)

        if "model" in st.session_state:
            fig = overlay_prediction_only(
                img_path=tmp_path,
                model=st.session_state.model,
                transform=get_val_transform(),
                threshold=threshold,
                alpha=alpha
            )
            st.pyplot(fig)
        else:
            st.warning("Сначала загрузите модель")

    # Отображение метрик из папки images/unet
    metrics_dir = "../images/unet"  # путь к вашей папке с картинками
    if os.path.isdir(metrics_dir):
        st.header("Метрики обучения")
        img_files = sorted(
            f for f in os.listdir(metrics_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
        )
        for img_name in img_files:
            img_path = os.path.join(metrics_dir, img_name)
            caption = os.path.splitext(img_name)[0]
            # выводим каждую картинку отдельно, сверху вниз, с увеличенной шириной
            st.image(
                Image.open(img_path),
                caption=caption,
                use_container_width=False,
                width=800  # задаёт ширину в пикселях
            )
    else:
        st.warning(f"Папка с метриками не найдена: {metrics_dir}")

if __name__ == "__main__":
    main()