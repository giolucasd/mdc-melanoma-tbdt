import timm

from src.models.base import BaseModel


class EfficientNetV2Wrapper(BaseModel):
    """
    Wrapper for EfficientNet-V2 (via timm).
    Designed for binary or multi-class classification.
    """

    def __init__(self, config: dict = None):
        super().__init__(config)

        model_name = config.get("model_name", "tf_efficientnetv2_s")
        pretrained = config.get("pretrained", True)
        num_classes = config.get("num_classes", 1)
        freeze_encoder = config.get("freeze_encoder", True)
        unfreeze_encoder_blocks = config.get("unfreeze_encoder_blocks", 0)

        # Create model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        if unfreeze_encoder_blocks > 0:
            self.unfreeze_last_blocks(unfreeze_encoder_blocks)
        elif freeze_encoder:
            self.freeze_encoder()

    def forward(self, x):
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Freezing logic
    # ------------------------------------------------------------------

    def freeze_encoder(self):
        """Freeze all layers except the classifier head."""
        for name, param in self.backbone.named_parameters():
            if "classifier" in name or "head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_blocks(self, n_blocks: int):
        """
        Unfreeze the last n EfficientNet-V2 blocks.

        EfficientNet-V2 structure in timm:
        - self.backbone.blocks : list of stages
        """
        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Always keep classifier trainable
        for param in self.backbone.get_classifier().parameters():
            param.requires_grad = True

        blocks = self.backbone.blocks
        total_blocks = len(blocks)
        n_blocks = min(n_blocks, total_blocks)

        for i in range(total_blocks - n_blocks, total_blocks):
            for param in blocks[i].parameters():
                param.requires_grad = True

        print(f"Unfroze last {n_blocks} EfficientNet-V2 blocks out of {total_blocks}.")
