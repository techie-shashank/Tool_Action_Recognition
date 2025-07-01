import argparse
import shutil
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data.dataset import ToolTrackingWindowDataset
from data.loader import ToolTrackingDataLoader
from sklearn.preprocessing import LabelEncoder
from logger import configure_logger, logger
from models.utils import get_model_class
from semi_supervised.train import train_semi_supervised
from utils import remove_undefined_class
from utils import config, config_path, train_model, get_percentage_of_data, FocalLoss
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


def load_and_preprocess_data(tool, sensors):
    logger.info("Loading and preprocessing data...")
    data_loader = ToolTrackingDataLoader(source=r"./../data/tool-tracking-data")
    Xt, y, classes = data_loader.load_and_process(tool, sensors)
    Xt, y = remove_undefined_class(Xt, y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    dataset = ToolTrackingWindowDataset(Xt, y)
    return dataset, le


def split_data(dataset, train_ratio=0.7, val_ratio=0.15):
    torch.manual_seed(42)
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    data_ratio = config['data_ratio']
    train_dataset = get_percentage_of_data(train_dataset, data_ratio)
    logger.info(f"Dataset split into Train: {len(train_dataset)}, Val: {val_size}, Test: {test_size}")
    return train_dataset, val_dataset, test_dataset


def setup_model(model_name, dataset, le, device):
    sample_X, _ = dataset[0]
    time_steps = sample_X.shape[0]
    input_channels = sample_X.shape[1]
    num_classes = len(le.classes_)
    logger.info(f"Input Channels: {input_channels}, Time Steps: {time_steps}, Number of Classes: {num_classes}")
    model_class = get_model_class(model_name)
    model = model_class(input_channels, time_steps, num_classes).to(device)
    return model


def train(model_name, tool_name, sensor_name, experiment_dir):
    batch_size = config["batch_size"]
    epochs = config["epochs"]

    shutil.copy(config_path, os.path.join(experiment_dir, "config.json"))

    model_path = os.path.join(experiment_dir, "model.pt")

    # Data loading and preprocessing
    dataset, le = load_and_preprocess_data(tool_name, sensor_name)
    # Data splitting
    train_dataset, val_dataset, test_dataset = split_data(dataset)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setup_model(model_name, dataset, le, device)

    y_train = dataset.y.cpu().numpy()
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=0.5, reduction='mean')


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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        logger.info(f"Starting training for model: {model_name.upper()}")
        train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            num_epochs=epochs,
            save_dir=experiment_dir,
            model_name=model_name,
            tool_name=tool_name,
            sensor_name=sensor_name if isinstance(sensor_name, list) else [sensor_name]
        )

    # Save model
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    args = parse_arguments()
    experiments_dir = get_experiments_dir(args.model)
    log_path = os.path.join(experiments_dir, "train.log")
    configure_logger(log_path)
    train(args.model, args.tool, args.sensor, experiments_dir)
