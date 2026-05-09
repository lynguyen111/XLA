import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import config
from utils.visualize import plot_training_history
from training.evaluate import evaluate_model
from data.preprocess import get_dataloaders
from models.cnn import SimpleInsectCNN
from models.resnet import get_resnet50

from models.densenet import get_densenet121, get_densenet169, get_densenet201



def train_model(model_name="resnet"):
    print(
        f"Bắt đầu huấn luyện [{model_name.upper()}] trên thiết bị: {config.DEVICE.upper()}"
    )

    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print(f"Số lớp: {len(classes)} - {classes}")

    # Khởi tạo mô hình
    if model_name == "resnet":
        model = get_resnet50(num_classes=len(classes))

    elif model_name == "densenet121":
        model = get_densenet121(num_classes=len(classes))
    elif model_name == "densenet169":
        model = get_densenet169(num_classes=len(classes))
    elif model_name == "densenet201":
        model = get_densenet201(num_classes=len(classes))
    else:
        model = SimpleInsectCNN(num_classes=len(classes))
    model = model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    best_val_loss = float("inf")
    early_stop_counter = 0
    models_saved_dir = os.path.join(config.PROJECT_ROOT, "models_saved")
    os.makedirs(models_saved_dir, exist_ok=True)
    save_path = os.path.join(models_saved_dir, f"bestv2_{model_name}.pth")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

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
                print(
                    f"   + Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / total
        train_acc = correct / total

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
        val_acc = correct / total

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"🟢 Epoch {epoch:02d}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"   ⭐ Saved best model -> {save_path}")
        else:
            early_stop_counter += 1
            print(
                f"   ⏳ Early stopping: {early_stop_counter}/{config.EARLY_STOPPING_PATIENCE}"
            )
            if early_stop_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n[!] Early stopping tại epoch {epoch}!")
                break

    print("\n[✓] HOÀN TẤT HUẤN LUYỆN!")

    plot_training_history(history, model_name, models_saved_dir)

    # ---- EVALUATE ----
    model.load_state_dict(torch.load(save_path, map_location=config.DEVICE))
    evaluate_model(model, test_loader, test_loader.dataset, classes, models_saved_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="resnet",
        choices=[
            "cnn",
            "resnet",
            "mobilenetv2",
            "mobilenetv3",
            "densenet121",
            "densenet169",
            "densenet201",
            "efficientnet_b3",
        ],
    )
    args = parser.parse_args()
    train_model(args.model)
