import argparse
import os
import random

import torch
import numpy as np

from logger import configure_logger, logger
from data.preprocessing import split_data, preprocess_signals
from test import load_model, evaluate_model
from utils import load_data
from train import train, get_experiments_dir, balance_data
from utils import load_config

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["tcn", "fcn", "lstm"], help="Model type")
    parser.add_argument("--tool", type=str, required=True, help="Tool to filter data")
    parser.add_argument("--sensor", type=str, nargs='+', default='all', help="List of sensors to filter data")
    return parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_and_test(model_name, tool_name, sensor_name):
    config = load_config()
    set_seed()
    print(config)
    experiments_dir = get_experiments_dir(model_name)

    log_path = os.path.join(experiments_dir, "main.log")
    configure_logger(log_path)

    data_balancing = config.get("data_balancing", [])
    Xt, y, le = load_data(tool_name, sensor_name)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(Xt, y, data_ratio=config['data_ratio'])
    X_train, y_train = balance_data(X_train, y_train, data_balancing)
    X_train, X_val, X_test = preprocess_signals(X_train, X_val, X_test)

    # Train the model
    train(model_name, X_train, y_train, X_val, y_val, le, experiments_dir)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(le.classes_)
    time_steps = X_test[0].shape[0]
    input_channels = X_test[0].shape[1]
    saved_model_path = os.path.join(experiments_dir, "model.pt")
    model = load_model(model_name, input_channels, time_steps, num_classes, saved_model_path, device)

    # Run test
    evaluate_model(model, X_test, y_test, device, le, experiments_dir)


if __name__ == "__main__":
    args = parse_arguments()
    model_name = args.model
    tool_name = args.tool
    sensor_name = args.sensor

    train_and_test(model_name, tool_name, sensor_name)