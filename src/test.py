import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from data.loader import ToolTrackingDataLoader
from data.dataset import ToolTrackingWindowDataset
from sklearn.preprocessing import LabelEncoder
from logger import configure_logger, logger
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from models.utils import get_model_class
from data.preprocessing import split_data, preprocess_signals
from utils import remove_undefined_class, load_data


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["tcn", "fcn", "lstm"], help="Model type")
    parser.add_argument("--tool", type=str, required=True, help="Tool to filter data")
    parser.add_argument("--sensor", type=str, nargs='+', default='all', help="List of sensors to filter data")
    return parser.parse_args()


def load_model(model_name, input_channels, time_steps, num_classes, saved_model_path, device):
    logger.info("Loading the model...")
    model = get_model_class(model_name)(input_channels, time_steps, num_classes).to(device)
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()
    logger.info("Model loaded and set to evaluation mode.")
    return model

def calculate_store_metrics(y_true, y_pred, save_dir="metrics_results", labels=None, class_names=None):
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
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=4)

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


def evaluate_model(model, X_test, y_test, device, le, save_dir, config=None):
    logger.info("Starting model evaluation...")

    if config is None:
        config = {}
    test_dataset = ToolTrackingWindowDataset(X_test, y_test, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 32))

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    target_names = le.classes_.astype(str).tolist()
    calculate_store_metrics(
        all_labels, all_preds, save_dir=f"{save_dir}/metrics_results",
        labels=np.arange(len(le.classes_)), class_names=target_names
    )


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

    # Load test data
    X_f, y_f, le = load_data(args.tool, args.sensor)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X_f, y_f)
    _, _, X_test = preprocess_signals(X_train, X_val, X_test)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(le.classes_)
    time_steps = X_test[0].shape[0]
    input_channels = X_test[0].shape[1]
    saved_model_path = os.path.join(run_dir, "model.pt")
    model = load_model(args.model, input_channels, time_steps, num_classes, saved_model_path, device)

    # Run test
    evaluate_model(model, X_test, y_test, device, le, run_dir)
