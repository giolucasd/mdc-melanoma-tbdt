import torch.nn as nn
from torchvision import models

from src.models.base import BaseModel


class ResNet50Wrapper(BaseModel):
    """Wraps torchvision ResNet50 and replaces the classifier head with a 1-logit output."""

    def __init__(self, config: dict = None):
        super().__init__(config)

        model_name = config.get("model_name", "resnet50")
        pretrained = config.get("pretrained", True)
        number_of_classes = config.get("num_classes", 1)
        freeze_encoder = config.get("freeze_encoder", False)

        # Load backbone
        self.backbone = getattr(models, model_name)(
            weights="DEFAULT" if pretrained else None
        )

        # Replace classifier head (fc)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, number_of_classes)

        if freeze_encoder:
            self.freeze_encoder()

    def forward(self, x):
        return self.backbone(x)

    def freeze_encoder(self):
        """Freeze encoder except the final classification layer."""
        for name, param in self.backbone.named_parameters():
            if name.startswith("fc"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True
