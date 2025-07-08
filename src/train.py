import argparse
import shutil
import os
from collections import Counter

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from data.dataset import ToolTrackingWindowDataset
from logger import configure_logger, logger
from models.utils import get_model_class
from semi_supervised.train import train_semi_supervised
from data.preprocessing import split_data, preprocess_signals, balance_data
from visualization.plots import plot_and_save_training_curves, visualize_channel_attention
from utils import load_data, get_weighted_sampler
from utils import load_config, config_path, train_model, FocalLoss
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["tcn", "fcn", "lstm"], help="Model type")
    parser.add_argument("--tool", type=str, required=True, help="Tool to filter data")
    parser.add_argument("--sensor", type=str, nargs='+', default='all', help="List of sensors to filter data")
    return parser.parse_args()


def get_experiments_dir(model_name):
    base_dir = os.path.join(r"./../experiments", model_name)
    os.makedirs(base_dir, exist_ok=True)

    # Get the highest run number
    existing_runs = [
        int(d.split("_")[-1]) for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    ]
    next_run = max(existing_runs, default=0) + 1

    # Create the new experiment directory
    experiment_dir = os.path.join(base_dir, f"run_{next_run}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def setup_model(model_name, time_steps, input_channels, num_classes, device):
    logger.info(f"Input Channels: {input_channels}, Time Steps: {time_steps}, Number of Classes: {num_classes}")
    model_class = get_model_class(model_name)
    model = model_class(input_channels, time_steps, num_classes).to(device)
    return model


def train(model_name, X_train, y_train, X_val, y_val, le, experiment_dir):
    config = load_config()
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    data_balancing = config.get("data_balancing", [])
    shutil.copy(config_path, os.path.join(experiment_dir, "config.json"))

    model_path = os.path.join(experiment_dir, "model.pt")

    train_dataset = ToolTrackingWindowDataset(X_train, y_train)
    val_dataset = ToolTrackingWindowDataset(X_val, y_val)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_X = X_train[0]
    time_steps = sample_X.shape[0]
    input_channels = sample_X.shape[1]
    num_classes = len(le.classes_)
    model = setup_model(model_name, time_steps, input_channels, num_classes, device)

    # Compute class weights
    labels = np.array(train_dataset.y)
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    alpha = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=1.6, reduction='mean') if 'focal_loss' in data_balancing else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Train
    if config["semi_supervised"]["active"]:
        logger.info(f"Starting Semi Supervised training for model: {model_name.upper()}")
        train_semi_supervised(
            model,
            train_dataset,
            val_dataset,
            criterion,
            optimizer,
            device,
            num_epochs=epochs)
    else:

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=get_weighted_sampler(
            labels
        )) if "weighted_sampling" in data_balancing else DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        all_sampled_labels = []

        # Collect labels for one epoch worth of data:
        for _, labels in train_loader:
            all_sampled_labels.extend(labels.cpu().numpy())

        print("Training sample label distribution (after 1 epoch):")
        print(Counter(all_sampled_labels))

        logger.info(f"Starting training for model: {model_name.upper()}")
        metrics = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=epochs)

        plot_and_save_training_curves(metrics, experiment_dir)
        if hasattr(model, 'channel_attention'):
            visualize_channel_attention(model, experiment_dir)

    # Save model
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    config = load_config()
    args = parse_arguments()
    experiments_dir = get_experiments_dir(args.model)
    log_path = os.path.join(experiments_dir, "train.log")
    configure_logger(log_path)

    # Data loading and preprocessing
    data_balancing = config.get("data_balancing", [])
    Xt, y, le = load_data(args.tool, args.sensor)
    (X_train, y_train), (X_val, y_val), (X_test, _) = split_data(Xt, y, data_ratio=config['data_ratio'])
    X_train, y_train = balance_data(X_train, y_train, data_balancing)
    X_train, X_val, _ = preprocess_signals(X_train, X_val, X_test)

    train(args.model, X_train, y_train, X_val, y_val, le, experiments_dir)
