import torch.nn as nn
from torchvision import models

from src.models.base import BaseModel


class EfficientNetWrapper(BaseModel):
    """Wraps torchvision EfficientNet and replaces the classifier head with a 1-logit output."""

    def __init__(self, config: dict = None):
        super().__init__()

        model_name = config.get("model_name", "efficientnet_b0")
        pretrained = config.get("pretrained", True)
        number_of_classes = config.get("num_classes", 1)
        freeze_encoder = config.get("freeze_encoder", False)

        self.backbone = getattr(models, model_name)(
            weights="DEFAULT" if pretrained else None
        )

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier[1] = nn.Linear(in_features, number_of_classes)

        if freeze_encoder:
            self.freeze_encoder()

    def forward(self, x):
        return self.backbone(x)

    def freeze_encoder(self):
        """Freeze encoder."""
        for name, param in self.backbone.named_parameters():
            if name.startswith("classifier"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True
