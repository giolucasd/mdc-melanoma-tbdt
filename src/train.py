import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score


class ModelTrainer:
    """Trainer for melanoma classification model.

    Accepts optional `pos_weight` to balance positive class when using
    `BCEWithLogitsLoss`. `pos_weight` can be a float or a 1-element tensor.
    """

    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu", pos_weight=None, lr=0.01):
        self.model = model.to(device)
        self.device = device
        # Configure criterion with optional pos_weight
        if pos_weight is not None:
            pw = pos_weight if isinstance(pos_weight, torch.Tensor) else torch.tensor(float(pos_weight), dtype=torch.float)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw.to(device))
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            all_preds.extend(predicted.flatten())
            all_labels.extend(labels.cpu().numpy().flatten().astype(int))
            
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = balanced_accuracy_score(all_labels, all_preds)
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validating")
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
                all_preds.extend(predicted.flatten())
                all_labels.extend(labels.cpu().numpy().flatten().astype(int))
                
                progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = balanced_accuracy_score(all_labels, all_preds)
        return avg_loss, avg_acc
    
    def fit(self, train_loader, val_loader, epochs=10):
        """Train the model for specified epochs."""
        print(f"Training on {self.device}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Step scheduler
            self.scheduler.step()
    
    def save_model(self, path):
        """Save model to file."""
        # Save state dict
        torch.save(self.model.state_dict(), path)

        # Also save training history (if available) next to the model
        try:
            import json
            import os
            history_path = f"{os.path.splitext(path)[0]}_history.json"
            # Convert any numpy types to Python native
            serializable = {k: [float(x) for x in v] for k, v in self.history.items()}
            with open(history_path, "w") as fh:
                json.dump(serializable, fh)
            print(f"Model and history saved to {path} and {history_path}")
        except Exception:
            print(f"Model saved to {path} (failed to save history)")
    
    def load_model(self, path):
        """Load model from file."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
