# FR-Photometric-Stereo

Multi-Task Face Recognition sử dụng dữ liệu Photometric Stereo (Albedo, Normal, Depth Map).

---

## Giới thiệu

Dự án nghiên cứu Multi-Task Learning cho Face Recognition, hỗ trợ các task: Face ID, Gender, Emotion, Pose, Facial Hair, Spectacles.

---

## Cấu trúc dự án

### Thư mục gốc

| File/Folder      | Mô tả                            |
| ---------------- | -------------------------------- |
| requirements.txt | Danh sách thư viện cần cài       |
| going_modular/   | Thư mục chứa toàn bộ source code |

---

### going_modular/data processing and cleaning/

Các notebook xử lý và chuẩn bị dữ liệu.

| File                                             | Mô tả                               |
| ------------------------------------------------ | ----------------------------------- |
| ALBEDO_PK_SAMPLER_ConvNextV2.ipynb               | Lấy mẫu và xử lý dữ liệu Albedo     |
| NORMALMAP_PK_SAMPLER_ConvNextV2.ipynb            | Lấy mẫu và xử lý dữ liệu Normal Map |
| Concat_ALBEDO_NORMAL_PK_SAMPLER_ConvNextV2.ipynb | Ghép dữ liệu Albedo + Normal        |
| Concat_PK_Supler.ipynb                           | Notebook hỗ trợ ghép dữ liệu        |
| albedo_sample.ipynb                              | Lấy mẫu dữ liệu Albedo đơn giản     |
| albedo_random.ipynb                              | Lấy mẫu ngẫu nhiên Albedo           |
| data_chuan_lan_cuoi.ipynb                        | Chuẩn hóa dữ liệu lần cuối          |

---

### going_modular/dataloader/

| File         | Mô tả                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------ |
| multitask.py | Dataset class cho single-modal và multi-modal (PhotometricDataset, ConcatCustomExrDataset) |

---

### going_modular/model/

| File                        | Mô tả                                                                      |
| --------------------------- | -------------------------------------------------------------------------- |
| MTLFaceRecognition.py       | Model MTL cho single-modal (1 loại ảnh)                                    |
| ConcatMTLFaceRecognition.py | Model MTL cho multi-modal (ghép 3 loại ảnh)                                |
| grl.py                      | Gradient Reversal Layer, đảo gradient khi backprop để học feature bất biến |

---

### going_modular/model/backbone/

| File                | Mô tả                                                  |
| ------------------- | ------------------------------------------------------ |
| convnext_v2_mifr.py | Backbone ConvNeXt V2 kết hợp Feature Separation Module |
| mifr.py             | Backbone ResNet với attention                          |
| irse.py             | Backbone IR-SE (Improved ResNet)                       |

---

### going_modular/model/head/

Các head cho từng task.

| File          | Mô tả                                            |
| ------------- | ------------------------------------------------ |
| id.py         | MagLinear head cho Face ID, dùng adaptive margin |
| gender.py     | Head phân loại giới tính                         |
| emotion.py    | Head phân loại cảm xúc                           |
| pose.py       | Head phân loại tư thế                            |
| facialhair.py | Head phát hiện râu                               |
| spectacles.py | Head phát hiện kính                              |

---

### going_modular/loss/

| File                   | Mô tả                                                   |
| ---------------------- | ------------------------------------------------------- |
| MultiTaskLoss.py       | Loss tổng hợp cho single-modal, tự động cân bằng weight |
| ConcatMultiTaskLoss.py | Loss tổng hợp cho multi-modal                           |
| WeightClassMagLoss.py  | MagFace loss có class weighting cho Face ID             |
| focalloss/FocalLoss.py | Focal Loss cho dữ liệu mất cân bằng                     |

---

### going_modular/train_eval/

| File            | Mô tả                              |
| --------------- | ---------------------------------- |
| train.py        | Vòng lặp training cho single-modal |
| concat_train.py | Vòng lặp training cho multi-modal  |

---

### going_modular/utils/

| File                                 | Mô tả                                                      |
| ------------------------------------ | ---------------------------------------------------------- |
| ExperimentManager.py                 | Quản lý experiment, tạo thư mục, lưu config, ghi log       |
| metrics.py                           | Hiển thị progress, tính toán metrics                       |
| roc_auc.py                           | Tính AUC cho các task                                      |
| transforms.py                        | Custom augmentation (GaussianNoise, RandomResizedCropRect) |
| ModelCheckPoint.py                   | Lưu model theo epoch và best metric                        |
| MultiMetricEarlyStopping.py          | Early stopping theo nhiều metric                           |
| PolynomialLRWarmup.py                | Learning rate scheduler polynomial warmup                  |
| WarmupCosineAnnealingWarmRestarts.py | Learning rate scheduler cosine annealing                   |

---

## Cài đặt

1. Clone repo
2. Tạo virtual environment
3. Cài requirements.txt
4. Cài thêm PyTorch, timm, pandas, scikit-learn, tabulate, tensorboard

---

## Định dạng dữ liệu

CSV chứa các cột: id, session, Gender, Spectacles, Facial_Hair, Pose, Emotion

Dữ liệu ảnh lưu dạng .npy trong thư mục theo cấu trúc: data/{id}/{session}/

---

## Tham khảo

- MagFace (CVPR 2021)
- ConvNeXt V2 (CVPR 2023)
- Multi-Task Learning Using Uncertainty to Weigh Losses (CVPR 2018)
- Focal Loss (ICCV 2017)
- Domain-Adversarial Training (JMLR 2016)

---

## Sơ đồ kiến trúc

### Kiến trúc tổng quan

```
Input (112x112)                         Output
     |                                     |
     v                                     v
+-----------+                    +------------------+
|  Albedo   |                    |    Face ID       |
|  Normal   | --> Backbone -->   |    Gender        |
|  Depth    |     (ConvNeXt)     |    Emotion       |
+-----------+                    |    Pose          |
                                 |    Facial Hair   |
                                 |    Spectacles    |
                                 +------------------+
```

### Pipeline xử lý

```
                    +---------------------------+
                    |     Input Images          |
                    | (Albedo, Normal, Depth)   |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |   ConvNeXt V2 Backbone    |
                    |   (Feature Extraction)    |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    | Feature Separation Module |
                    |          (FSM)            |
                    +-------------+-------------+
                                  |
          +-----------+-----------+-----------+-----------+
          |           |           |           |           |
          v           v           v           v           v
     +--------+  +--------+  +--------+  +--------+  +--------+
     |  ID    |  | Gender |  |Emotion |  |  Pose  |  | Specs  |
     |  Head  |  |  Head  |  |  Head  |  |  Head  |  |  Head  |
     +--------+  +--------+  +--------+  +--------+  +--------+
          |           |           |           |           |
          v           v           v           v           v
     +--------+  +--------+  +--------+  +--------+  +--------+
     |MagFace |  | Focal  |  | Focal  |  | Focal  |  | Focal  |
     | Loss   |  |  Loss  |  |  Loss  |  |  Loss  |  |  Loss  |
     +--------+  +--------+  +--------+  +--------+  +--------+
          |           |           |           |           |
          +-----------+-----------+-----------+-----------+
                                  |
                                  v
                    +---------------------------+
                    |    Multi-Task Loss        |
                    |  (Auto Loss Weighting)    |
                    +---------------------------+
```

### Multi-Modal Fusion (Concat)

```
+----------+     +----------+     +----------+
|  Normal  |     |  Albedo  |     |  Depth   |
|   MTL    |     |   MTL    |     |   MTL    |
+----+-----+     +----+-----+     +----+-----+
     |                |                |
     |   +------------+------------+   |
     |   |                         |   |
     +---+-----------+-------------+---+
                     |
                     v
              +-------------+
              |   Concat    |
              |  (512 x 3)  |
              +------+------+
                     |
                     v
              +-------------+
              | Final Heads |
              |   (1536)    |
              +-------------+
```

### Cấu trúc thư mục

```
FR_Photometric_stereo/
|
+-- requirements.txt
+-- README.md
|
+-- going_modular/
    |
    +-- data processing and cleaning/
    |   +-- *.ipynb (notebooks xử lý dữ liệu)
    |
    +-- dataloader/
    |   +-- multitask.py
    |
    +-- model/
    |   +-- MTLFaceRecognition.py
    |   +-- ConcatMTLFaceRecognition.py
    |   +-- grl.py
    |   |
    |   +-- backbone/
    |   |   +-- convnext_v2_mifr.py
    |   |   +-- mifr.py
    |   |   +-- irse.py
    |   |
    |   +-- head/
    |       +-- id.py, gender.py, emotion.py
    |       +-- pose.py, facialhair.py, spectacles.py
    |
    +-- loss/
    |   +-- MultiTaskLoss.py
    |   +-- ConcatMultiTaskLoss.py
    |   +-- WeightClassMagLoss.py
    |   +-- focalloss/FocalLoss.py
    |
    +-- train_eval/
    |   +-- train.py
    |   +-- concat_train.py
    |
    +-- utils/
        +-- ExperimentManager.py
        +-- metrics.py, roc_auc.py
        +-- transforms.py
        +-- ModelCheckPoint.py
        +-- MultiMetricEarlyStopping.py
        +-- PolynomialLRWarmup.py
        +-- WarmupCosineAnnealingWarmRestarts.py
```
