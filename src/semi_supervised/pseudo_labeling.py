import numpy as np

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, ConcatDataset
from logger import logger
from utils import config, train_model
import os
import json

from utils import get_weighted_sampler

config_path = os.path.join(r'../', "config.json")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

def generate_pseudo_labels(model, unlabeled_loader, device, threshold=None):
    if threshold is None:
        semi_config = config.get("semi_supervised", {})
        threshold = semi_config.get("threshold", 0.9)
        print(f"Threshold is set to value: {threshold}")
    model.eval()
    pseudo_X, pseudo_y = [], []

    logger.info("Generating pseudo-labels for unlabeled data...")

    with torch.no_grad():
        for i, batch in enumerate(unlabeled_loader):
            # Handle case where batch is a tuple (X,) or (X, None)
            X_batch = batch[0].to(device)

            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            mask = confs > threshold
            num_selected = mask.sum().item()
            logger.debug(f"Batch {i+1}: Selected {num_selected} samples above threshold.")

            if num_selected > 0:
                pseudo_X.append(X_batch[mask].cpu())
                pseudo_y.append(preds[mask].cpu())

    if pseudo_X:
        pseudo_X = torch.cat(pseudo_X, dim=0)
        pseudo_y = torch.cat(pseudo_y, dim=0)
        logger.info(f"Generated {len(pseudo_X)} high-confidence pseudo-labeled samples.")
        return TensorDataset(pseudo_X, pseudo_y)
    else:
        logger.warning("No pseudo-labels met the confidence threshold.")
        return None

def train_pseudo_labelling(model, labeled_loader, unlabeled_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    train_model(model, labeled_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)

    # Generate pseudo-labels
    logger.info("Starting pseudo-labeling process...")
    pseudo_dataset = generate_pseudo_labels(model, unlabeled_loader, device, threshold=config["semi_supervised"].get("threshold"))

    # Combine and retrain
    if pseudo_dataset is not None:
        logger.info("Combining labeled and pseudo-labeled datasets for retraining...")
        combined_dataset = ConcatDataset([labeled_loader.dataset, pseudo_dataset])

        data_balancing = config.get('data_balancing', [])
        labels = [labeled_loader.dataset.dataset.y[idx].item() for idx in labeled_loader.dataset.indices] + pseudo_dataset.tensors[1].tolist()
        combined_loader = DataLoader(combined_dataset, batch_size=64, sampler=get_weighted_sampler(
            labels
        )) if "weighted_sampling" in data_balancing else DataLoader(
            combined_dataset, batch_size=64, shuffle=True
        )

        logger.info("Retraining on combined dataset...")
        train_model(model, combined_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)
    else:
        logger.info("Skipping retraining due to no confident pseudo-labels.")