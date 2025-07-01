import json
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.utils.data import random_split
from logger import logger
import shutil

# Load configuration
config_path = os.path.join(r'../', "config.json")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)


def ensure_model_exists(model_type, saved_model_path):
    if os.path.exists(saved_model_path):
        logger.info(f"Model already exists at {saved_model_path}")
        return

    os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)

    experiment_root = os.path.join("../experiments", model_type)
    if not os.path.exists(experiment_root):
        raise FileNotFoundError(f"No experiment directory found for model: {model_type}")

    run_dirs = sorted(
        [os.path.join(experiment_root, d) for d in os.listdir(experiment_root)
         if os.path.isdir(os.path.join(experiment_root, d)) and d.startswith("run_")],
        reverse=True
    )

    for run_dir in run_dirs:
        candidate_model_path = os.path.join(run_dir, "model.pt")
        if os.path.exists(candidate_model_path):
            shutil.copy(candidate_model_path, saved_model_path)
            logger.info(f"Copied model from {candidate_model_path} to {saved_model_path}")
            return

    raise FileNotFoundError(f"No model.pt found in any runs under {experiment_root}")

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10,
                save_dir="./", model_name="Model", tool_name="Tool", sensor_name="Sensors"):

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        logger.info(f"[Epoch {epoch+1}] "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | "
                    f"Val Acc: {val_acc:.2f}%")

    # --- Plotting ---
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title(f"Loss over Epochs\n{model_name.upper()} - {tool_name} - {', '.join(sensor_name)}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Val Accuracy", color="green")
    plt.title(f"Validation Accuracy over Epochs\n{model_name.upper()} - {tool_name} - {', '.join(sensor_name)}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(plot_path)
    logger.info(f"Saved training curves to {plot_path}")



def remove_undefined_class(Xt, y):
    mask = y != 8
    X_filtered = Xt[mask]
    y_filtered = y[mask]
    return X_filtered, y_filtered


def get_percentage_of_data(dataset, percentage):
    """
    Returns a subset of the dataset based on the specified percentage.
    """
    total_size = len(dataset)
    subset_size = int(percentage * total_size)
    _, subset = random_split(dataset, [total_size - subset_size, subset_size])
    return subset

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
        # CE = -log(p_t) * alpha
        # FL = (1 - p_t)^gamma * CE
        # where p_t is the model's estimated probability for each class and alpha is a weighting factor for each class.
        # alpha is used to balance the importance of different classes using class weights.

        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        # can have sum..

        return focal_loss
