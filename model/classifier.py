import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
