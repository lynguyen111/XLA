import os
import torch

# --- PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
TRAIN_DIR = os.path.join(DATA_DIR, 'dataset', 'train')
VAL_DIR = os.path.join(DATA_DIR, 'dataset', 'val')
TEST_DIR = os.path.join(DATA_DIR, 'dataset', 'test')

# --- DATASET SPLIT ---
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --- HYPERPARAMETERS ---
EPOCHS = 20
LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# --- DEVICE ---
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# --- CLASSES ---
# Những lớp mà chúng ta huấn luyện
CLASSES = [
    "Kien", "Ong", "Buom", "Chuon_chuon", "Chau_chau", 
    "Muoi", "Gian", "Bo_rua", "Bo_canh_cung", "Sau_buom",
    "Ruoi", "Nhen", "Ong_bap_cay"
]
NUM_CLASSES = len(CLASSES)
