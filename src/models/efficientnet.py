import torch.nn as nn
from torchvision import models


def get_efficientnet_b3(num_classes=13):
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
