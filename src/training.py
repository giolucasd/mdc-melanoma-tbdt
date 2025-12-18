import math

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: raw model outputs (no sigmoid)
        # targets: {0,1} labels

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p = sigmoid(logits)
        p = torch.sigmoid(logits)

        # focal weight
        pt = p * targets + (1 - p) * (1 - targets)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        # bce = - torch.log(pt)

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


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

        if config.get("loss_fn") == "BCEWithLogitsLoss":
            pos_weight = float(config.get("bce_pos_weight", 1.0))
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        elif config.get("loss_fn") == "FocalLoss":
            focal_alpha = float(config.get("focal_alpha", 0.25))
            focal_gamma = float(config.get("focal_gamma", 2.0))
            self.loss_fn = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
            )
        else:
            raise ValueError(f"Unsupported loss function: {config.get('loss_fn')}")

        self.train_preds = []
        self.train_targets = []

        self.val_preds = []
        self.val_targets = []

        self.beta = float(config.get("fbeta_beta", 2))

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

        # -------- METRICS --------
        acc = accuracy_score(targets, preds)
        bal_acc = balanced_accuracy_score(targets, preds)
        precision = precision_score(targets, preds, zero_division=0)
        recall = recall_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)
        fbeta = fbeta_score(targets, preds, beta=self.beta, zero_division=0)

        # -------- LOGGING --------
        self.log("train_acc", acc)
        self.log("train_bal_acc", bal_acc, prog_bar=True)
        self.log("train_precision", precision)
        self.log("train_recall", recall)
        self.log("train_f1", f1)
        self.log("train_fbeta", fbeta)

        # -------- CONFUSION MATRIX --------
        cm = confusion_matrix(targets, preds, normalize="true")

        fig = plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0.0, vmax=1.0)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Train Confusion Matrix")

        self.logger.experiment.add_figure(
            "train_confusion_matrix",
            fig,
            self.current_epoch,
        )
        plt.close(fig)

        # -------- CLEAN UP --------
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

        # -------- METRICS --------
        acc = accuracy_score(targets, preds)
        bal_acc = balanced_accuracy_score(targets, preds)
        precision = precision_score(targets, preds, zero_division=0)
        recall = recall_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)
        fbeta = fbeta_score(targets, preds, beta=self.beta, zero_division=0)

        # -------- LOGGING --------
        self.log("val_acc", acc)
        self.log("val_bal_acc", bal_acc, prog_bar=True)
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)
        self.log("val_fbeta", fbeta)

        # -------- CONFUSION MATRIX --------
        cm = confusion_matrix(targets, preds, normalize="true")

        fig = plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0.0, vmax=1.0)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        # Loga a imagem no TensorBoard
        self.logger.experiment.add_figure(
            "val_confusion_matrix",
            fig,
            self.current_epoch,
        )
        plt.close(fig)

        # -------- CLEAN UP --------
        self.val_preds.clear()
        self.val_targets.clear()

    def predict_step(self, batch, batch_idx):
        images, ids = batch
        logits = self(images).squeeze(1)
        return {"logits": logits, "ids": ids}

    def configure_optimizers(self):
        learning_rate = float(self.hparams.get("learning_rate", 1e-3))
        weight_decay = float(self.hparams.get("weight_decay", 1e-4))
        warmup_epochs = int(self.hparams.get("warmup_epochs", 5))
        max_epochs = int(self.hparams.get("max_epochs", 50))

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
