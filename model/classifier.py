import timm
import torch.nn as nn


class SkinLesionClassifier(nn.Module):
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(1792),
            nn.Dropout(0.4),
            nn.Linear(1792, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(num_classes: int = 8, pretrained: bool = True) -> nn.Module:
    return SkinLesionClassifier(num_classes=num_classes, pretrained=pretrained)
