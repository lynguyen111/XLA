import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
from data.preprocess import get_dataloaders
from models.cnn import SimpleInsectCNN
from models.resnet import get_resnet50


def train_model(model_name="resnet"):
    print(f"Bắt đầu huấn luyện [{model_name.upper()}] trên thiết bị: {config.DEVICE.upper()}")

    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print(f"Số lớp: {len(classes)} - {classes}")

    # Khởi tạo mô hình
    if model_name == "resnet":
        model = get_resnet50(num_classes=len(classes), freeze_backbone=False)
    else:
        model = SimpleInsectCNN(num_classes=len(classes))
    model = model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss = float("inf")
    models_saved_dir = os.path.join(config.PROJECT_ROOT, "models_saved")
    os.makedirs(models_saved_dir, exist_ok=True)
    save_path = os.path.join(models_saved_dir, f"best_{model_name}.pth")

    for epoch in range(1, config.EPOCHS + 1):
        # ---- TRAIN ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f"   + Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        train_loss = running_loss / total
        train_acc  = correct / total * 100

        # ---- VALIDATION ----
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= total
        val_acc   = correct / total * 100

        scheduler.step(val_loss)

        print(f"🟢 Epoch {epoch:02d}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"   ⭐ Saved best model -> {save_path}")

    print("\n[✓] HOÀN TẤT HUẤN LUYỆN!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet", choices=["cnn", "resnet"])
    args = parser.parse_args()
    train_model(args.model)
