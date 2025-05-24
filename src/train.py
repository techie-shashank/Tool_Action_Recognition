import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data.dataset import ToolTrackingWindowDataset
from data.loader import ToolTrackingDataLoader
from models.fcn import FCNClassifier
from fhgutils import filter_labels, one_label_per_window
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Parse command line args
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name")
args = parser.parse_args()

# Create a new experiment folder with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = os.path.join(r"./../experiments", args.model, f"run_{timestamp}")
os.makedirs(experiment_dir, exist_ok=True)

# Set paths
log_path = os.path.join(experiment_dir, "training.log")
model_path = os.path.join(experiment_dir, "model.pt")


# ---------------------- Set Up Logging ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------- Training Function ----------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        logger.info(f"[Epoch {epoch+1}] "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | "
                    f"Val Acc: {val_acc:.2f}%")

# ---------------------- Data Loading and Preprocessing ----------------------
logger.info("Loading and preprocessing data...")
data_loader = ToolTrackingDataLoader(source=r"./../data/tool-tracking-data")
Xt, Xc, y, classes = data_loader.load_and_process(tool="electric_screwdriver", desc_filter='acc')
Xt_f, Xc_f, y_f = filter_labels(labels=[-1], Xt=Xt, Xc=Xc, y=y)

y_f = one_label_per_window(y=y_f)
le = LabelEncoder()
y_f = le.fit_transform(y_f)

dataset = ToolTrackingWindowDataset(Xt_f, y_f)

# ---------------------- Data Splitting ----------------------
torch.manual_seed(42)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
logger.info(f"Dataset split into Train: {train_size}, Val: {val_size}, Test: {test_size}")

# ---------------------- DataLoaders ----------------------
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ---------------------- Model Setup ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_X, _ = dataset[0]
time_steps = sample_X.shape[0]
input_channels = sample_X.shape[1]
num_classes = len(le.classes_)

model = FCNClassifier(input_channels, time_steps, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

logger.info("Starting training...")
train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)

# ---------------------- Optional: Save Model ----------------------
torch.save(model.state_dict(), model_path)
logger.info(f"Model saved to {model_path}")
