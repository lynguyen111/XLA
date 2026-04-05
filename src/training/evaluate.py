import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
from utils.visualize import plot_confusion_matrix, plot_sample_predictions


def evaluate_model(model, test_loader, test_dataset, classes, save_dir):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    print(
        f"\n[TEST] Accuracy: {acc:.4f} ({(all_preds == all_labels).sum()}/{len(all_labels)})"
    )

    plot_confusion_matrix(all_labels, all_preds, classes, save_dir)
    plot_sample_predictions(model, test_dataset, classes, save_dir)


if __name__ == "__main__":
    import argparse
    from data.preprocess import get_dataloaders
    from models.resnet import get_resnet50
    from models.cnn import SimpleInsectCNN

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet", choices=["cnn", "resnet"])
    args = parser.parse_args()

    _, _, test_loader, classes = get_dataloaders()

    if args.model == "resnet":
        model = get_resnet50(num_classes=len(classes))
    else:
        model = SimpleInsectCNN(num_classes=len(classes))

    models_saved_dir = os.path.join(config.PROJECT_ROOT, "models_saved")
    save_path = os.path.join(models_saved_dir, f"bestv2_{args.model}.pth")
    model.load_state_dict(torch.load(save_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)

    evaluate_model(model, test_loader, test_loader.dataset, classes, models_saved_dir)
