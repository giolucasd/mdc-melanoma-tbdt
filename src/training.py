import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


class MelanomaLitModule(pl.LightningModule):
    """
    Model agnostic LightningModule for melanoma classification.

    Configurable training hyperparameters via config dict.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(config)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_preds = []
        self.train_targets = []

        self.val_preds = []
        self.val_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images).squeeze(1)

        preds = (torch.sigmoid(logits) > 0.5).long()
        self.train_preds.append(preds.cpu())
        self.train_targets.append(targets.cpu())

        loss = self.loss_fn(logits, targets.float())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        preds = torch.cat(self.train_preds).numpy()
        targets = torch.cat(self.train_targets).numpy()

        bal_acc = balanced_accuracy_score(targets, preds)

        self.log("train_balanced_accuracy", bal_acc, prog_bar=True)

        self.train_preds.clear()
        self.train_targets.clear()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images).squeeze(1)

        loss = self.loss_fn(logits, targets.float())
        preds = (torch.sigmoid(logits) > 0.5).long()

        self.val_preds.append(preds.cpu())
        self.val_targets.append(targets.cpu())

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds).numpy()
        targets = torch.cat(self.val_targets).numpy()

        bal_acc = balanced_accuracy_score(targets, preds)
        self.log("val_balanced_accuracy", bal_acc, prog_bar=True)

        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        learning_rate = self.hparams.get("learning_rate", 1e-3)
        weight_decay = self.hparams.get("weight_decay", 1e-4)
        warmup_epochs = self.hparams.get("warmup_epochs", 5)
        max_epochs = self.hparams.get("max_epochs", 50)

        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return float(current_epoch) / float(max(1, warmup_epochs))

            progress = (current_epoch - warmup_epochs) / float(
                max(1, max_epochs - warmup_epochs)
            )

            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * math.pi)))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
