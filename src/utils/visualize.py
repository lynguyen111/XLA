import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_training_history(history, model_name, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"training_curve_{model_name}.png")
    plt.savefig(plot_path)
    print(f"[✓] Đã lưu biểu đồ -> {plot_path}")
    plt.show()
    plt.close("all")


def plot_confusion_matrix(all_labels, all_preds, classes, save_dir):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(len(classes) + 2, len(classes) + 2))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path)
    print(f"[✓] Đã lưu confusion matrix -> {path}")
    plt.show()
    plt.close("all")


def plot_sample_predictions(model, test_dataset, classes, save_dir, n=8):
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import config

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    indices = random.sample(range(len(test_dataset)), n)
    model.eval()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, true_label = test_dataset[idx]
            output = model(img_tensor.unsqueeze(0).to(config.DEVICE))
            probs = torch.softmax(output, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_label].item() * 100

            # Denormalize để hiển thị
            img = img_tensor.numpy().transpose(1, 2, 0)
            img = std * img + mean
            img = np.clip(img, 0, 1)

            color = "green" if pred_label == true_label else "red"
            axes[i].imshow(img)
            axes[i].set_title(
                f"Thật: {classes[true_label]}\nDự đoán: {classes[pred_label]}\n{confidence:.1f}%",
                color=color, fontsize=9
            )
            axes[i].axis("off")

    plt.suptitle("8 ảnh ngẫu nhiên từ tập test", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "sample_predictions.png")
    plt.savefig(path)
    print(f"[✓] Đã lưu sample predictions -> {path}")
    plt.show()
    plt.close("all")
