import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Đường dẫn project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
from data.preprocess import get_dataloaders
from models.cnn import SimpleInsectCNN

def train_model():
    print(f"Bắt đầu huấn luyện mô hình. Đang chạy trên thiết bị (Device): {config.DEVICE.upper()}")
    
    # Kéo bộ nạp dữ liệu (Loaders)
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print(f"Tổng quan - Số lớp: {len(classes)} - {classes}")
    
    # Khởi tạo mô hình CNN Custom
    model = SimpleInsectCNN(num_classes=len(classes))
    model = model.to(config.DEVICE)
    
    # Định nghĩa Loss (Hàm Mất Mát) & Optimizer (Thuật toán tối ưu)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Lặp qua các chu kì (Epochs)
    best_val_loss = float('inf')
    
    for epoch in range(1, config.EPOCHS + 1):
        # ========= TIỀN ĐẠO (TRAINING) =========
        model.train() # Chuyển model sang chế độ train (Bật Dropout, BatchNorm)
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad() # Xoá bộ nhớ gradient vòng trước
            
            outputs = model(inputs) # Dự đoán
            loss = criterion(outputs, labels) # Tính sai số
            
            loss.backward() # Lan truyền ngược tìm đường tối ưu
            optimizer.step() # Cập nhật tham số mô hình
            
            # Ghi nhận kết quả
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1) # Lấy class có xác suất cao nhất
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Print tiến độ mỗi 50 steps
            if (batch_idx + 1) % 50 == 0:
                print(f"   + Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        # Tính Loss và Acc trên toàn bộ tập Train
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100
        
        # ========= KIỂM CHỨNG (VALIDATION) =========
        model.eval() # Tắt Dropout, cố định BatchNorm
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        # Đóng băng gradient lúc validate cho nhanh và tiết kiệm RAM
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100
        
        print(f"🟢 Hết Epoch {epoch:02d}/{config.EPOCHS} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")
        
        # Lưu lại model tốt nhất nếu Validation Loss thấp kỷ lục
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            
            models_saved_dir = os.path.join(config.PROJECT_ROOT, "models_saved")
            if not os.path.exists(models_saved_dir):
                 os.makedirs(models_saved_dir)
                 
            save_path = os.path.join(models_saved_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f" -> ⭐ Đã lưu Checkpoint (Best Validation Loss) tại {save_path}")

    print("\n[✓] HOÀN TẤT QUÁ TRÌNH HUẤN LUYỆN DỮ LIỆU CẢM BIẾN!")
    
if __name__ == "__main__":
    train_model()
