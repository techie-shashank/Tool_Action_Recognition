import argparse
import os
from logger import configure_logger, logger
from train import train, get_experiments_dir
from test import test


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["tcn", "fcn", "lstm"], help="Model type")
    parser.add_argument("--tool", type=str, required=True, help="Tool to filter data")
    parser.add_argument("--sensor", type=str, nargs='+', default='all', help="List of sensors to filter data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    tool_name = args.tool
    sensor_name = args.sensor
    model_name = args.model
    experiments_dir = get_experiments_dir(model_name)

    log_path = os.path.join(experiments_dir, "main.log")
    configure_logger(log_path)

    # Train the model
    train(model_name, tool_name, sensor_name, experiments_dir)

    # Test the model
    test(model_name, tool_name, sensor_name, experiments_dir)
