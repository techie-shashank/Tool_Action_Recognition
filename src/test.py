import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from data.loader import ToolTrackingDataLoader
from data.dataset import ToolTrackingWindowDataset
from models.fcn import FCNClassifier
from models.lstm import LSTMClassifier
from sklearn.preprocessing import LabelEncoder
from fhgutils import filter_labels, one_label_per_window
from src.logger import configure_logger, logger
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from src.models.utils import get_model_class


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["semi_fcn", "semi_lstm", "fcn", "lstm"], help="Model type")
    parser.add_argument("--tool", type=str, required=True, help="Tool to filter data")
    parser.add_argument("--sensor", type=str, required=True, help="Sensor filter for data")
    return parser.parse_args()

def load_and_preprocess_data(tool, sensor, data_loader_class, filter_labels, one_label_per_window):
    logger.info("Loading and preprocessing data...")
    data_loader = data_loader_class(source=r"./../data/tool-tracking-data")
    Xt, Xc, y, classes = data_loader.load_and_process(tool=tool, desc_filter=sensor)
    Xt_f, Xc_f, y_f = filter_labels(labels=[-1], Xt=Xt, Xc=Xc, y=y)
    X_f = Xt_f[:, :, 1:]
    y_f = one_label_per_window(y=y_f)
    le = LabelEncoder()
    y_f = le.fit_transform(y_f)
    logger.info("Data loaded and preprocessed successfully.")
    return X_f, y_f, le

def split_dataset(X_f, y_f, dataset_class, train_ratio=0.7, val_ratio=0.15):
    logger.info("Splitting dataset into train, validation, and test sets...")
    dataset = dataset_class(X_f, y_f)
    torch.manual_seed(42)
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    logger.info(f"Dataset split completed: Train={train_size}, Val={val_size}, Test={test_size}")
    return test_dataset

def create_test_loader(test_dataset, batch_size=32):
    logger.info("Creating DataLoader for the test dataset...")
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    logger.info("Test DataLoader created successfully.")
    return test_loader

def load_model(model_name, dataset, le, saved_model_path, device, model_classes):
    logger.info("Loading the model...")
    sample_X, _ = dataset[0]
    time_steps = sample_X.shape[0]
    input_channels = sample_X.shape[1]
    num_classes = len(le.classes_)
    model = get_model_class(model_name)(input_channels, time_steps, num_classes).to(device)
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()
    logger.info("Model loaded and set to evaluation mode.")
    return model

def calculate_store_metrics(y_true, y_pred, save_dir="metrics_results", class_names=None):
    """
    Evaluate model predictions and save metrics.

    Args:
        y_true (np.ndarray): True labels.
        y_prob (np.ndarray): Predicted probabilities.
        threshold (float): Threshold for binary classification.
        save_dir (str): Directory to save the metrics results.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    print(f"\n[RESULT] Accuracy: {acc:.4f}")
    print(f"[RESULT] Precision (macro): {precision:.4f}")
    print(f"[RESULT] Recall (macro): {recall:.4f}")
    print(f"[RESULT] F1 Score (macro): {f1:.4f}")
    print("\n[RESULT] Classification Report:\n", report)

    # ---------------------- Confusion Matrix ----------------------
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix")
    plt.tight_layout()

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"confusion_matrix.png"))
    with open(os.path.join(save_dir, f"classification_report.txt"), 'w') as f:
        f.write(report)

    logger.info(f"Confusion matrix and classification report saved to {save_dir}")


def evaluate_model(model, test_loader, device, le, save_dir):
    logger.info("Starting model evaluation...")

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    target_names = [str(cls) for cls in le.classes_]
    calculate_store_metrics(all_labels, all_preds, save_dir=f"{save_dir}/metrics_results", class_names=target_names)


def test(model_name, tool_name, sensor_name, run_dir):
    logger.info("Starting the test process...")
    model_classes = {
        "fcn": FCNClassifier,
        "lstm": LSTMClassifier
    }

    X_f, y_f, le = load_and_preprocess_data(tool_name, sensor_name, ToolTrackingDataLoader, filter_labels,
                                            one_label_per_window)
    test_dataset = split_dataset(X_f, y_f, ToolTrackingWindowDataset)
    test_loader = create_test_loader(test_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_model_path = os.path.join(run_dir, "model.pt")
    model = load_model(model_name, test_dataset, le, saved_model_path, device, model_classes)

    evaluate_model(model, test_loader, device, le, run_dir)


if __name__ == "__main__":
    args = parse_arguments()

    run_number = None
    if run_number:
        run_dir = os.path.join(
            os.path.join(r"./../experiments", args.model),
            f"run_{run_number}"
        )
    else:
        run_dir = os.path.join(r"./../saved_model", args.model)

    log_path = os.path.join(run_dir, "test.log")
    configure_logger(log_path)
    test(args.model, args.tool, args.sensor, run_dir)
