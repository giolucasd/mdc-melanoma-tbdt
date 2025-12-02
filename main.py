import os
import sys

# MUST be set BEFORE importing any scientific libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path
from matplotlib.widgets import Button
from src.data import get_train_val_dataloaders, MelanomaDataset, DATA_PATH
from src.model import SimpleCNN
from src.train import ModelTrainer


def analyze_dataset():
    """Analyze and visualize dataset properties."""

    # Load data
    train_csv = DATA_PATH / "train.csv"
    val_csv = DATA_PATH / "val.csv"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Load dataloaders
    train_loader, val_loader = get_train_val_dataloaders(batch_size=32, num_workers=4)

    # Create dataset objects to get samples
    root_dir = DATA_PATH
    train_dataset = MelanomaDataset(train_df, root_dir)
    val_dataset = MelanomaDataset(val_df, root_dir)

    # Calculate statistics
    train_size = len(train_df)
    val_size = len(val_df)
    train_healthy = int((train_df["target"] == 0).sum())
    train_melanoma = int((train_df["target"] == 1).sum())
    val_healthy = int((val_df["target"] == 0).sum())
    val_melanoma = int((val_df["target"] == 1).sum())

    # Check balance
    train_ratio = train_melanoma / train_size if train_size > 0 else 0
    val_ratio = val_melanoma / val_size if val_size > 0 else 0
    balance_threshold = 0.45
    is_balanced = (balance_threshold < train_ratio < (1 - balance_threshold)) and \
                  (balance_threshold < val_ratio < (1 - balance_threshold))

    # Get sample images
    healthy_idx = train_df[train_df["target"] == 0].index[0]
    melanoma_idx = train_df[train_df["target"] == 1].index[0]

    healthy_img, _ = train_dataset[healthy_idx]
    melanoma_img, _ = train_dataset[melanoma_idx]

    # Create visualization
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Dataset Analysis - Melanoma Classification", fontsize=16, fontweight="bold")

    # 1. Dataset sizes
    ax1 = plt.subplot(2, 3, 1)
    categories = ["Train", "Val"]
    sizes = [train_size, val_size]
    colors = ["#3498db", "#e74c3c"]
    bars = ax1.bar(categories, sizes, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
    ax1.set_ylabel("Number of Images", fontsize=11, fontweight="bold")
    ax1.set_title("Dataset Partition Sizes", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f"{int(size)}", ha="center", va="bottom", fontweight="bold")

    # 2. Train class distribution
    ax2 = plt.subplot(2, 3, 2)
    train_classes = [train_healthy, train_melanoma]
    labels = ["Healthy (0)", "Melanoma (1)"]
    colors_pie = ["#2ecc71", "#e74c3c"]
    wedges, texts, autotexts = ax2.pie(train_classes, labels=labels, autopct="%1.1f%%",
                                        colors=colors_pie, startangle=90, textprops={"fontsize": 10})
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
    ax2.set_title(f"Train Distribution (n={train_size})", fontsize=12, fontweight="bold")

    # 3. Val class distribution
    ax3 = plt.subplot(2, 3, 3)
    val_classes = [val_healthy, val_melanoma]
    wedges, texts, autotexts = ax3.pie(val_classes, labels=labels, autopct="%1.1f%%",
                                        colors=colors_pie, startangle=90, textprops={"fontsize": 10})
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
    ax3.set_title(f"Val Distribution (n={val_size})", fontsize=12, fontweight="bold")

    # 4. Healthy example
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(healthy_img)
    ax4.set_title("Healthy (Label: 0)", fontsize=11, fontweight="bold", color="green")
    ax4.axis("off")

    # 5. Melanoma example
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(melanoma_img)
    ax5.set_title("Melanoma (Label: 1)", fontsize=11, fontweight="bold", color="red")
    ax5.axis("off")

    # 6. Balance status and statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    balance_status = "✓ BALANCED" if is_balanced else "✗ IMBALANCED"
    balance_color = "green" if is_balanced else "orange"

    stats_text = f"""
    DATASET STATISTICS
    {'=' * 40}

    TRAIN PARTITION:
      • Total: {train_size} images
      • Healthy: {train_healthy} ({train_ratio*100:.1f}%)
      • Melanoma: {train_melanoma} ({(1-train_ratio)*100:.1f}%)

    VALIDATION PARTITION:
      • Total: {val_size} images
      • Healthy: {val_healthy} ({val_ratio*100:.1f}%)
      • Melanoma: {val_melanoma} ({(1-val_ratio)*100:.1f}%)

    {'=' * 40}
    BALANCE STATUS: {balance_status}
    """

    ax6.text(0.1, 0.5, stats_text, fontsize=10, family="monospace",
            verticalalignment="center", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    # Add a button to load a saved model and display its metrics/curves
    plt.subplots_adjust(bottom=0.12)
    button_ax = fig.add_axes([0.4, 0.02, 0.2, 0.05])
    load_button = Button(button_ax, 'Load saved model')

    # Prepare val_loader for evaluation if needed
    _, val_loader_local = get_train_val_dataloaders(batch_size=32, num_workers=4)

    def model_summary_text(model):
        lines = [str(model), "\nParameters:"]
        total = 0
        for name, p in model.named_parameters():
            lines.append(f"{name}: {tuple(p.shape)} | {p.numel()} params")
            total += p.numel()
        lines.append(f"Total params: {total}")
        return "\n".join(lines)

    def on_load(event):
        import tkinter as tk
        from tkinter import filedialog
        import os, json

        root = tk.Tk()
        root.withdraw()

        initial = os.path.abspath(os.path.join(os.getcwd(), "models"))
        file_path = filedialog.askopenfilename(title="Select model file", initialdir=initial,
                                               filetypes=[("PyTorch", "*.pth *.pt"), ("All files", "*.*")])
        if not file_path:
            return

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SimpleCNN()
        try:
            model.load_state_dict(torch.load(file_path, map_location=device))
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"Failed to load model: {e}")
            return

        # Try to load history sidecar
        hist_path = os.path.splitext(file_path)[0] + "_history.json"
        history = None
        if os.path.exists(hist_path):
            try:
                with open(hist_path, "r") as fh:
                    history = json.load(fh)
            except Exception:
                history = None

        # Prepare model summary text early
        model_info = model_summary_text(model)

        # If history is missing, we will not retrain here; proceed to evaluate on validation set

        # Build a single new window (figure) that contains model info, curves and ROC/PR together
        import sklearn.metrics as skm
        import warnings
        warnings.filterwarnings("ignore")

        # Create 2-column layout: left column for model architecture, right column for plots
        fig2 = plt.figure(figsize=(16, 10))
        fig2.suptitle("Loaded Model - Architecture, Training Curves & ROC/PR", fontsize=14, fontweight="bold")

        gs = fig2.add_gridspec(3, 2, width_ratios=[1, 1.5], height_ratios=[1, 1, 0.8])
        ax_info = fig2.add_subplot(gs[:, 0])         # left column: full height for model info
        ax_acc = fig2.add_subplot(gs[0, 1])          # top right: accuracy vs epoch
        ax_roc = fig2.add_subplot(gs[1, 1])          # middle right: ROC
        ax_pr = fig2.add_subplot(gs[2, 1])           # bottom right: PR

        # If history exists, plot balanced accuracy per epoch inside ax_acc
        metrics_text = ""
        if history is not None and ("val_acc" in history or "train_acc" in history):
            epochs = range(1, len(history.get("val_acc", history.get("train_acc", []))) + 1)
            if "train_acc" in history:
                ax_acc.plot(epochs, history["train_acc"], label="Train Balanced Acc", marker="o")
            if "val_acc" in history:
                ax_acc.plot(epochs, history["val_acc"], label="Val Balanced Acc", marker="o")
            ax_acc.set_xlabel("Epoch")
            ax_acc.set_ylabel("Balanced Accuracy")
            ax_acc.set_title("Balanced Accuracy per Epoch")
            ax_acc.legend()
            ax_acc.grid(True)

            metrics_text = f"Loaded history from: {hist_path}\n"
            if "train_acc" in history:
                metrics_text += f"Final Train Balanced Acc: {history['train_acc'][-1]:.4f}\n"
            if "val_acc" in history:
                metrics_text += f"Final Val Balanced Acc: {history['val_acc'][-1]:.4f}\n"
        else:
            # No history: evaluate model on validation set to compute balanced accuracy and collect probs for ROC/PR
            all_probs = []
            all_labels = []
            with torch.no_grad():
                for images, labels in val_loader_local:
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    all_probs.extend(probs.tolist())
                    all_labels.extend(labels.numpy().flatten().astype(int).tolist())
            try:
                bal_acc = skm.balanced_accuracy_score(all_labels, [1 if p>0.5 else 0 for p in all_probs])
            except Exception:
                bal_acc = None

            # Show the single-point metric as a bar + text in ax_acc
            ax_acc.bar([0], [bal_acc if bal_acc is not None else 0], color="#2ecc71")
            ax_acc.set_ylim(0, 1)
            ax_acc.set_xticks([])
            ax_acc.set_ylabel("Balanced Accuracy")
            ax_acc.set_title("Validation Balanced Accuracy (evaluated)")
            if bal_acc is not None:
                ax_acc.text(0, bal_acc + 0.03, f"{bal_acc:.4f}", ha="center", fontsize=12, fontweight="bold")

            metrics_text = f"Evaluated Val Balanced Acc: {bal_acc:.4f}\n" if bal_acc is not None else "Metrics: N/A\n"

        # Compute ROC and PR using validation predictions (if not yet computed above)
        try:
            if history is not None and ("val_acc" in history or "train_acc" in history):
                # still compute ROC/PR from validation set
                pass
            # Ensure we have probs/labels
            probs, labels_for_metrics = None, None
            if 'all_probs' in locals():
                probs = np.array(all_probs)
                labels_for_metrics = np.array(all_labels)
            else:
                # compute now
                probs = []
                labels_for_metrics = []
                with torch.no_grad():
                    for images, labels in val_loader_local:
                        images = images.to(device)
                        outputs = model(images)
                        probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten().tolist())
                        labels_for_metrics.extend(labels.numpy().flatten().astype(int).tolist())
                probs = np.array(probs)
                labels_for_metrics = np.array(labels_for_metrics)

            if probs is not None and len(probs) > 0:
                fpr, tpr, _ = skm.roc_curve(labels_for_metrics, probs)
                roc_auc = skm.auc(fpr, tpr)
                precision, recall, _ = skm.precision_recall_curve(labels_for_metrics, probs)
                ap = skm.average_precision_score(labels_for_metrics, probs)

                ax_roc.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
                ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title("ROC Curve")
                ax_roc.legend()

                ax_pr.plot(recall, precision, label=f"PR (AP={ap:.3f})")
                ax_pr.set_xlabel("Recall")
                ax_pr.set_ylabel("Precision")
                ax_pr.set_title("Precision-Recall Curve")
                ax_pr.legend()
            else:
                ax_roc.text(0.5, 0.5, "No validation predictions available", ha="center")
                ax_pr.text(0.5, 0.5, "No validation predictions available", ha="center")
        except Exception as e:
            ax_roc.text(0.5, 0.5, f"ROC failure: {e}", ha="center")
            ax_pr.text(0.5, 0.5, f"PR failure: {e}", ha="center")

        # Show full model architecture in left column with metrics at top
        ax_info.axis("off")
        
        # Build detailed architecture text with metrics header
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        header = f"MODEL: {model.__class__.__name__}\n"
        header += f"Total Parameters: {total_params:,}\n"
        header += f"Trainable Parameters: {trainable_params:,}\n"
        header += f"{metrics_text.strip()}\n"
        header += "=" * 50 + "\n\n"
        
        full_info = header + model_info
        
        # Display in scrollable text area (using monospace for alignment)
        text_obj = ax_info.text(0.02, 0.98, full_info, fontsize=8, family="monospace", 
                               verticalalignment="top", horizontalalignment="left",
                               bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.8))

        # Add toggle button to collapse/expand model info by hiding the entire ax_info
        toggle_ax = fig2.add_axes([0.01, 0.96, 0.08, 0.03])
        toggle_button = Button(toggle_ax, 'Hide Info')
        
        # Use a mutable container to track state
        toggle_state = {'visible': True}

        def toggle_info_callback(event):
            toggle_state['visible'] = not toggle_state['visible']
            ax_info.set_visible(toggle_state['visible'])
            toggle_button.label.set_text('Show Info' if not toggle_state['visible'] else 'Hide Info')
            fig2.canvas.draw_idle()

        toggle_button.on_clicked(toggle_info_callback)

        fig2.tight_layout(rect=[0, 0.01, 1, 0.96])
        fig2.show()

        # Optionally update the original analysis pane (ax6) with a brief note
        ax6.clear()
        ax6.axis("off")
        ax6.text(0.1, 0.5, f"Model loaded:\n{os.path.basename(file_path)}", fontsize=10, family="monospace",
                 verticalalignment="center")
        fig.canvas.draw_idle()

    load_button.on_clicked(on_load)

    plt.show()


def main():
    """Analyze dataset or train model (GPU-optimized)."""
    print("=" * 50)
    print("Melanoma Classification - Main Menu")
    print("=" * 50)
    print("Choose an option:")
    print("1. Analyze Dataset")
    print("2. Train Model (GPU-accelerated) - Interactive")
    print()
    print("TIP: For reproducible training with configs, use:")
    print("  python -m scripts.train --config configs/baseline_cnn_no_augmentation.yaml --output-suffix exp_name")
    print("=" * 50)
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        analyze_dataset()
    elif choice == "2":
        # Load configuration from YAML
        config_path = "configs/baseline_cnn_no_augmentation.yaml"
        
        print(f"Loading configuration from {config_path}...")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Extract hyperparameters from config
        batch_size = config["training"]["batch_size"]
        num_workers = config["training"]["num_workers"]
        epochs = config["training"]["max_epochs"]
        
        print(f"Hyperparameters loaded from config:")
        print(f"  - batch_size: {batch_size}")
        print(f"  - num_workers: {num_workers}")
        print(f"  - epochs: {epochs}")
        
        # Load data
        print("Loading data...")
        train_loader, val_loader = get_train_val_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        # Create model
        print("Creating model...")
        model = SimpleCNN()
        
        # Create trainer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Compute pos_weight from training set to handle class imbalance
        train_csv = DATA_PATH / "train.csv"
        train_df = pd.read_csv(train_csv)
        neg = int((train_df["target"] == 0).sum())
        pos = int((train_df["target"] == 1).sum())
        if pos == 0:
            pos_weight = 1.0
        else:
            pos_weight = neg / pos
        print(f"Training set: {neg} negative, {pos} positive. pos_weight={pos_weight:.3f}")

        trainer = ModelTrainer(model, device=device, pos_weight=pos_weight)
        
        # Train model
        print("Starting training...")
        trainer.fit(train_loader, val_loader, epochs=epochs)
        
        # Save model
        trainer.save_model("models/baseline_cnn.pth")
        
        # Print final results
        print("\nTraining completed!")
        print(f"Final Train Accuracy: {trainer.history['train_acc'][-1]:.4f}")
        print(f"Final Val Accuracy: {trainer.history['val_acc'][-1]:.4f}")
    else:
        print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
