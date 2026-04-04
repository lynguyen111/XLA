import torch.nn as nn
from torchvision import models


def get_resnet50(num_classes=13, freeze_backbone=False):
    """
    ResNet50 pretrained ImageNet.
    - freeze_backbone=True: chỉ train classifier head (nhanh, ít overfit)
    - freeze_backbone=False: fine-tune toàn bộ (chính xác hơn)
    """
    model = models.resnet50(weights=None)  # train từ đầu, không dùng pretrained

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Thay classifier head cuối phù hợp số lớp
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )

    return model
