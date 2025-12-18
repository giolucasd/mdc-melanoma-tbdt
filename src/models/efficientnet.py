import torch.nn as nn
from torchvision import models

from src.models.base import BaseModel


class EfficientNetWrapper(BaseModel):
    """Wraps torchvision EfficientNet and replaces the classifier head with a 1-logit output."""

    def __init__(self, config: dict = None):
        super().__init__(config)

        model_name = config.get("model_name", "efficientnet_b0")
        pretrained = config.get("pretrained", True)
        number_of_classes = config.get("num_classes", 1)
        freeze_encoder = config.get("freeze_encoder", True)
        unfreeze_encoder_blocks = config.get("unfreeze_encoder_blocks", 0)

        self.backbone = getattr(models, model_name)(
            weights="DEFAULT" if pretrained else None
        )

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier[1] = nn.Linear(in_features, number_of_classes)

        if unfreeze_encoder_blocks > 0:
            self.unfreeze_last_blocks(unfreeze_encoder_blocks)
        elif freeze_encoder:
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

    def unfreeze_last_blocks(self, n_blocks: int):
        """Unfreeze the last `n_blocks` blocks of the encoder."""
        # First freeze everything again
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

        # Always keep classifier trainable
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

        # Unfreeze last n_blocks from features
        total_blocks = len(self.backbone.features)
        n_blocks = min(n_blocks, total_blocks)  # clamp

        for i in range(total_blocks - n_blocks, total_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True

        print(
            f"Unfroze last {n_blocks} feature blocks "
            f"out of {total_blocks} total blocks."
        )
