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
        freeze_encoder = config.get("freeze_encoder", True)
        unfreeze_encoder_blocks = config.get("unfreeze_encoder_blocks", 0)

        # Load backbone
        self.backbone = getattr(models, model_name)(
            weights="DEFAULT" if pretrained else None
        )

        # Replace classifier head (fc)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, number_of_classes)

        if unfreeze_encoder_blocks > 0:
            self.unfreeze_last_blocks(unfreeze_encoder_blocks)
        elif freeze_encoder:
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

    def unfreeze_last_blocks(self, n_blocks: int):
        """Unfreeze the last `n_blocks` of the encoder."""
        # Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Always keep classifier trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

        # Ordered blocks
        blocks = [
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ]

        total = len(blocks)
        n = min(n_blocks, total)

        # Unfreeze last n blocks
        for block in blocks[total - n :]:
            for param in block.parameters():
                param.requires_grad = True

        print(f"Unfroze last {n} blocks (from layer{5 - n} to layer4).")
