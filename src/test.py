import os
import argparse
import torch
import logging
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from data.loader import ToolTrackingDataLoader
from data.dataset import ToolTrackingWindowDataset
from models.fcn import FCNClassifier
from sklearn.preprocessing import LabelEncoder
from fhgutils import filter_labels, one_label_per_window


# ---------------------- Set Up Logging ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler('./../logs/test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Parse command line args
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name")
args = parser.parse_args()

saved_model_path = os.path.join("./../saved_model", args.model, f"model.pt")


# Load and preprocess data
data_loader = ToolTrackingDataLoader(source=r"./../data/tool-tracking-data")
Xt, Xc, y, classes = data_loader.load_and_process(tool="electric_screwdriver", desc_filter='acc')
Xt_f, Xc_f, y_f = filter_labels(labels=[-1], Xt=Xt, Xc=Xc, y=y)
y_f = one_label_per_window(y=y_f)
le = LabelEncoder()
y_f = le.fit_transform(y_f)

# Create dataset and split
dataset = ToolTrackingWindowDataset(Xt_f, y_f)
torch.manual_seed(42)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size
_, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# DataLoader
test_loader = DataLoader(test_dataset, batch_size=32)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_X, _ = dataset[0]
time_steps = sample_X.shape[0]
input_channels = sample_X.shape[1]
num_classes = len(le.classes_)
model = FCNClassifier(input_channels, time_steps, num_classes).to(device)
model.load_state_dict(torch.load(saved_model_path))
model.eval()

# Evaluate model
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")
