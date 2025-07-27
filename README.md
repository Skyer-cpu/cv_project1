# Фаза 2 • Неделя • Четверг 
## Компьютерное зрение • Computer Vision
### Computer vision project • Yolo team 

- В соответствии с [инструкцией](https://github.com/Elbrus-DataScience/ds-phase-2/blob/master/08-nn/md/directory_structure.md) создайте git-репозиторий `cv_project` и добавьте туда членов команды. 
- Разработайте [multipage](https://blog.streamlit.io/introducing-multipage-apps/)-приложение с использованием [streamlit](streamlit.io):
  - Детекция [лиц](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset) с помощью любой версии YOLO c последующей маскировкой детектированной области ([пример](SCR-20240807-kgpq.png))
  - Детекция объектов с помощью [YOLOv11](https://docs.ultralytics.com/yolov1/tutorials/train_custom_data/#13-prepare-dataset-for-yolov5): [датасет](https://www.kaggle.com/datasets/davidbroberts/brain-tumor-object-detection-datasets) для детекции опухулей мозга по фотографии (для начала выбрать папку axial_t1wce_2_class)
  - ✳️ Примените модель [Unet](https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3) к задаче семантической сегментации [аэрокосмических снимков](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation)
 
 
**Требования к сервису:**

1. Все страницы должны поддерживать загрузку пользователем сразу нескольких файлов.
2. Страница с детекцией объектов должна поддерживать подгрузку файла по прямой ссылке.
3. Все страницы должны иметь раздел с информацией о моделях, качестве и процессе обучения:
   * Число эпох обучения
   * Объем выборок
   * Метрики (для детекции mAP, график PR кривой, confusion matrix и т.д.)


**Рекомендации**

- Сохрайняте модели в процессе обучения каждые несколько эпох, это позволит сделать быстрый прототип сервиса, после этого нужно будет просто заменить файл с весами модели и все продолжит работать
- Отлаживайте модели локально: создайте сеть, реализуйете цикл обучения и проведите 1-3 эпохи обучения. Если ошибок не возникает, то загрузите блокнот в Google Colab, включите GPU и запустите обучение (так же используйте [https://www.kaggle.com/](https://www.kaggle.com/) или [https://vast.ai](https://vast.ai)
- Сохраненяйте модели на ваш Google Drive, тогда вы их не потеряете
- Не пытайтесь сделать все сразу, сконцентрируйтесь на конкретной задаче и доделайте её
- Не забывайте про канал группы и #resources – там уже есть довольно много ответов на ваши вопросы (включая репозитории старших коллег)
- Делайте пуши на гитхаб – как минимум в конце дня
- Созванивайтесь между собой!
- Если устали – сделайте перерыв, у вас достаточно времени на проект

> ❓[Как скачать данные с Kaggle в Google Colaboratory](https://github.com/Elbrus-DataScience/ds-phase-2/blob/master/08-nn/md/kaggle-colab.md)

> ❓[Как скачать данные с Google Drive в Google Colaboratory?](https://github.com/Elbrus-DataScience/ds-phase-2/blob/master/08-nn/md/drive-colab.md)
