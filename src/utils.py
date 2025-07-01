import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import random_split, WeightedRandomSampler
from logger import logger

import shutil

from data.loader import ToolTrackingDataLoader

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

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        total_train_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            l1_lambda = 0.5

            if hasattr(model, 'l1_regularization'):
                l1_penalty = model.l1_regularization()
            else:
                l1_penalty = 0.0

            loss = criterion(outputs, y_batch) + l1_lambda * l1_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            total_train_samples += X_batch.size(0)

        avg_train_loss = train_loss / total_train_samples
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        correct, total = 0, 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss / total
        val_acc = 100 * correct / total
        val_f1 = f1_score(all_targets, all_preds, average='macro')  # Macro F1 recommended for imbalanced classes

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        logger.info(f"[Epoch {epoch+1}] "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | "
                    f"Val Acc: {val_acc:.2f}%")

    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_f1_scores': val_f1_scores
    }
    return metrics


def remove_undefined_class(Xt, y):
    mask = y != 8
    X_filtered = Xt[mask]
    y_filtered = y[mask]
    return X_filtered, y_filtered


def load_data(tool, sensors):
    logger.info("Loading and preprocessing data...")
    data_loader = ToolTrackingDataLoader(source=r"./../data/tool-tracking-data")
    Xt, y, classes = data_loader.load_and_process(tool, sensors)
    Xt, y = remove_undefined_class(Xt, y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    return Xt, y, le

def get_percentage_of_data(dataset, percentage):
    """
    Returns a subset of the dataset based on the specified percentage.
    """
    total_size = len(dataset)
    subset_size = int(percentage * total_size)
    _, subset = random_split(dataset, [total_size - subset_size, subset_size])
    return subset


def get_weighted_sampler(labels):
    """
    Create a WeightedRandomSampler that samples classes inversely proportional
    to their frequency in the dataset to handle class imbalance.

    Args:
        labels (list or 1D tensor): List or tensor of integer class labels.

    Returns:
        WeightedRandomSampler: Sampler for balanced class sampling.
    """
    # If tensor, convert to list
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Count class occurrences
    label_counts = Counter(labels)

    # Compute weights for each class
    total_samples = len(labels)
    class_weights = {label: (total_samples / count) ** 0.75 for label, count in label_counts.items()}

    # Assign weights to each sample
    sample_weights = np.array([class_weights[label] for label in labels])

    # Debugging: Print class weights and sample weights
    print("Class Weights (scaled):", class_weights)
    print("Sample Weights (first 10):", sample_weights[:10])

    # Create the sampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler



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
