import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys

# Đảm bảo import được config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils import config
except ImportError:
    print("Vui lòng chạy từ thư mục dự án.")
    sys.exit(1)

def get_dataloaders(batch_size=None, image_size=None):
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if image_size is None:
        image_size = config.IMAGE_SIZE
        
    # --- Định nghĩa các Augmentation & Transform ---
    # Tập Train: Dùng các thao tác Random crop, lật ảnh ngang... để data xịn hơn
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), # Phóng to một chút
        transforms.RandomResizedCrop(image_size), # Cắt ngẫu nhiên xuống chuẩn 224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # Lắc nhẹ 15 độ
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Đổi ánh sáng
        transforms.ToTensor(), # Đưa về dạng tensor [0.0, 1.0]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Chuẩn hoá ImageNet
    ])

    # Tập Val và Test: Không áp dụng random augmentation, chỉ resize chuẩn và normalize (Tránh sai lệch lúc đánh giá)
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Đọc dataset bằng ImageFolder ---
    print(f"Loading data từ: {config.DATA_DIR}")
    train_dataset = datasets.ImageFolder(config.TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(config.VAL_DIR, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(config.TEST_DIR, transform=val_test_transform)
    
    # --- DataLoader ---
    # Dùng num_workers để tăng tốc đọc file song song
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, train_dataset.classes

if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print("Các nhãn được tìm thấy:", classes)
    
    # Kéo thử 1 batch
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)
