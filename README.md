# XLA – Nhận Diện Côn Trùng bằng Deep Learning

Ứng dụng phân loại ảnh côn trùng sử dụng các mô hình CNN (ResNet18, DenseNet121, EfficientNet-B3), kết hợp backend Flask và frontend React.

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Các lớp côn trùng](#các-lớp-côn-trùng)
- [Cài đặt môi trường](#cài-đặt-môi-trường)
- [Cấu trúc dataset](#cấu-trúc-dataset)
- [Huấn luyện mô hình](#huấn-luyện-mô-hình)
- [Chạy ứng dụng web](#chạy-ứng-dụng-web)
- [API](#api)
- [Cấu hình](#cấu-hình)

---

## Tổng quan

Dự án xây dựng pipeline hoàn chỉnh từ huấn luyện đến triển khai:

- **Training**: Script Python + Jupyter Notebook để train/evaluate các mô hình ResNet, DenseNet, EfficientNet
- **Backend**: Flask API nhận ảnh và trả về top-5 dự đoán
- **Frontend**: React (Vite) để upload ảnh và hiển thị kết quả nhận diện

---

## Cấu trúc dự án

```
XLA/
├── backend/
│   ├── app.py                  # Flask API server
│   └── requirements_backend.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/
│   ├── index.html
│   └── package.json
├── src/
│   ├── data/
│   │   └── preprocess.py       # DataLoader + augmentation
│   ├── models/
│   │   ├── resnet.py           # ResNet18/34/50/101
│   │   ├── densenet.py         # DenseNet121/169/201
│   │   └── efficientnet.py     # EfficientNet-B3
│   ├── training/
│   │   ├── train.py            # Vòng lặp huấn luyện
│   │   └── evaluate.py         # Đánh giá + confusion matrix
│   └── utils/
│       └── config.py           # Hyperparameter & path config
├── data/
│   └── dataset/
│       ├── train/
│       ├── val/
│       └── test/
├── models_saved/               # File .pth sau khi train
├── train_resnet.ipynb
├── train_densenet.ipynb
└── train_efficientnet.ipynb
```

---

## Các lớp côn trùng

| Tên file       | Tên hiển thị  |
|----------------|---------------|
| Bo_canh_cung   | Bọ Cánh Cứng  |
| Bo_rua         | Bọ Rùa        |
| Buom           | Bướm          |
| Chau_chau      | Châu Chấu     |
| Chuon_chuon    | Chuồn Chuồn   |
| Gian           | Gián          |
| Kien           | Kiến          |
| Muoi           | Muỗi          |
| Nhen           | Nhện          |
| Ong            | Ong           |
| Ong_bap_cay    | Ong Bắp Cày   |
| Ruoi           | Ruồi          |
| Sau_buom       | Sâu Bướm      |

---

## Cài đặt môi trường

### Yêu cầu

- Python 3.10+
- Node.js 18+

### Backend

```bash
# Tạo và kích hoạt virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# hoặc: .venv\Scripts\activate  # Windows

# Cài dependencies
pip install -r backend/requirements_backend.txt
```

### Frontend

```bash
cd frontend
npm install
```

---

## Cấu trúc dataset

Dataset cần được tổ chức theo cấu trúc `ImageFolder` của PyTorch:

```
data/dataset/
├── train/
│   ├── Kien/
│   │   ├── anh1.jpg
│   │   └── ...
│   ├── Buom/
│   └── ...
├── val/
│   ├── Kien/
│   └── ...
└── test/
    ├── Kien/
    └── ...
```

Tỉ lệ phân chia mặc định: **70% train / 15% val / 15% test**

---

## Huấn luyện mô hình

### Dùng script Python

```bash
cd src

# Train ResNet50
python training/train.py --model resnet

# Train DenseNet121
python training/train.py --model densenet121

# Train EfficientNet-B3
python training/train.py --model efficientnet_b3
```

Các tùy chọn `--model`: `cnn`, `resnet`, `densenet121`, `densenet169`, `densenet201`, `efficientnet_b3`

### Dùng Jupyter Notebook

Mở và chạy các notebook tương ứng:

- `train_resnet.ipynb` — ResNet18
- `train_densenet.ipynb` — DenseNet121
- `train_efficientnet.ipynb` — EfficientNet-B3

### Hyperparameters mặc định

| Tham số              | Giá trị |
|----------------------|---------|
| Epochs               | 70      |
| Batch size           | 32      |
| Learning rate        | 0.001   |
| Early stopping       | 8 epochs|
| Image size           | 224×224 |
| Optimizer            | Adam    |
| Scheduler            | ReduceLROnPlateau |

Model tốt nhất sẽ được lưu tại `models_saved/bestv2_<model_name>.pth`.

---

## Chạy ứng dụng web

### Bước 1: Khởi động Backend

```bash
source .venv/bin/activate
python backend/app.py
```

Backend chạy tại `http://localhost:5001`

> **Lưu ý**: File `backend/app.py` dòng 89 đang dùng đường dẫn tuyệt đối đến model. Cần cập nhật lại nếu chạy trên máy khác:
> ```python
> # Thay đường dẫn tuyệt đối bằng:
> os.path.join(MODELS_DIR, "best_resnet18.pth")
> ```

### Bước 2: Khởi động Frontend

```bash
cd frontend
npm run dev
```

Frontend chạy tại `http://localhost:3000`

Vite tự động proxy `/api/*` → `http://localhost:5001/*`

### Sử dụng

1. Mở trình duyệt tại `http://localhost:3000`
2. Chọn mô hình (ResNet18)
3. Upload ảnh côn trùng (kéo thả hoặc click)
4. Nhấn **Nhận Diện** để xem kết quả top-5

---

## API

### `GET /health`

Kiểm tra trạng thái server.

**Response:**
```json
{
  "status": "ok",
  "models": ["resnet18"],
  "device": "mps"
}
```

### `POST /predict`

Phân loại ảnh côn trùng.

**Request:** `multipart/form-data`
- `file`: file ảnh (JPG, PNG, ...)
- `model`: tên model (mặc định: `resnet18`)

**Response:**
```json
{
  "predictions": [
    {
      "rank": 1,
      "class_id": 6,
      "class_name": "Kien",
      "display_name": "Kiến",
      "emoji": "🐜",
      "confidence": 97.45
    }
  ],
  "model": "resnet18",
  "device": "mps"
}
```

---

## Cấu hình

Chỉnh sửa tại [src/utils/config.py](src/utils/config.py):

```python
EPOCHS = 70
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 8
IMAGE_SIZE = (224, 224)
```

Device được tự động phát hiện theo thứ tự: `cuda` → `mps` (Apple Silicon) → `cpu`
