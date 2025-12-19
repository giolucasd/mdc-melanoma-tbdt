import timm

from src.models.base import BaseModel


class ConvNeXtV2Wrapper(BaseModel):
    """ConvNeXt V2 wrapper for binary or multi-class classification."""

    def __init__(self, config: dict = None):
        super().__init__(config)

        model_name = config.get("model_name", "convnextv2_tiny")
        pretrained = config.get("pretrained", True)
        num_classes = config.get("num_classes", 1)
        freeze_encoder = config.get("freeze_encoder", True)
        unfreeze_encoder_blocks = config.get("unfreeze_encoder_blocks", 0)

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

    def freeze_encoder(self):
        """Freeze all except classifier head."""
        for name, param in self.backbone.named_parameters():
            if "head" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze entire model."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_blocks(self, n_blocks: int):
        """
        Unfreeze last n ConvNeXt stages.
        ConvNeXt V2 has 4 stages.
        """
        # Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Head always trainable
        for param in self.backbone.get_classifier().parameters():
            param.requires_grad = True

        stages = self.backbone.stages
        total_stages = len(stages)
        n_blocks = min(n_blocks, total_stages)

        for i in range(total_stages - n_blocks, total_stages):
            for param in stages[i].parameters():
                param.requires_grad = True

        print(f"Unfroze last {n_blocks} ConvNeXt stages out of {total_stages}.")
