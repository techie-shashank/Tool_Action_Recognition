import argparse
import json
import os
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import LabelEncoder

from data.loader import ToolTrackingDataLoader
from data.dataset import ToolTrackingWindowDataset
from models.fcn import FCNClassifier
from fhgutils import filter_labels, one_label_per_window

# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name (e.g., fcn)")
parser.add_argument("--tool", type=str, required=True, help="Tool to evaluate")
parser.add_argument("--sensor", type=str, required=True, help="Sensor type")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (model.pt)")
args = parser.parse_args()

# ---------------------- Load Config ----------------------
config_path = os.path.join(r'../', "config.json")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

batch_size = config["batch_size"]

# ---------------------- Device ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Data Loading ----------------------
print("[INFO] Loading and preprocessing data...")
data_loader = ToolTrackingDataLoader(source=r"./../data/tool-tracking-data")
Xt, Xc, y, classes = data_loader.load_and_process(tool=args.tool, desc_filter=args.sensor)
Xt_f, Xc_f, y_f = filter_labels(labels=[-1], Xt=Xt, Xc=Xc, y=y)
y_f = one_label_per_window(y=y_f)

le = LabelEncoder()
y_f = le.fit_transform(y_f)

# Remove timestamp
X_f = Xt_f[:, :, 1:]

dataset = ToolTrackingWindowDataset(X_f, y_f)

# Split data
torch.manual_seed(42)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

_, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ---------------------- Model Setup ----------------------
sample_X, _ = dataset[0]
time_steps = sample_X.shape[0]
input_channels = sample_X.shape[1]
num_classes = len(le.classes_)

model = FCNClassifier(input_channels, time_steps, num_classes).to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.eval()

# ---------------------- Evaluation ----------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

# ---------------------- Metrics ----------------------
acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
target_names = [str(cls) for cls in le.classes_]
report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)

print(f"\n[RESULT] Accuracy: {acc:.4f}")
print(f"[RESULT] Precision (macro): {precision:.4f}")
print(f"[RESULT] Recall (macro): {recall:.4f}")
print(f"[RESULT] F1 Score (macro): {f1:.4f}")
print("\n[RESULT] Classification Report:\n", report)

# ---------------------- Confusion Matrix ----------------------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix: {args.tool}")
plt.tight_layout()

# Save results
results_dir = os.path.join("./../evaluations", args.model)
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, f"confusion_matrix_{args.tool}.png"))
with open(os.path.join(results_dir, f"classification_report_{args.tool}.txt"), 'w') as f:
    f.write(report)

print(f"\n[INFO] Confusion matrix and classification report saved to {results_dir}")