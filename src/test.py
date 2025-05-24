import os
import argparse
import torch
import logging
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
parser.add_argument("--tool", type=str, required=True, help="Tool to filter data")
parser.add_argument("--sensor", type=str, required=True, help="Sensor filter for data")
args = parser.parse_args()

saved_model_path = os.path.join(r"./../saved_model", args.model, f"model.pt")


# Load and preprocess data
logger.info("Loading and preprocessing data...")
data_loader = ToolTrackingDataLoader(source=r"./../data/tool-tracking-data")
Xt, Xc, y, classes = data_loader.load_and_process(tool=args.tool, desc_filter=args.sensor)
Xt_f, Xc_f, y_f = filter_labels(labels=[-1], Xt=Xt, Xc=Xc, y=y)
y_f = one_label_per_window(y=y_f)
le = LabelEncoder()
y_f = le.fit_transform(y_f)
logger.info("Data loaded and preprocessed successfully.")

# Create dataset and split
logger.info("Splitting dataset into train, validation, and test sets...")
dataset = ToolTrackingWindowDataset(Xt_f, y_f)
torch.manual_seed(42)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size
_, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])
logger.info(f"Dataset split completed: Train={train_size}, Val={val_size}, Test={test_size}")


# DataLoader
logger.info("Creating DataLoader for the test dataset...")
test_loader = DataLoader(test_dataset, batch_size=32)
logger.info("Test DataLoader created successfully.")


# Load model
logger.info("Loading the model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_X, _ = dataset[0]
time_steps = sample_X.shape[0]
input_channels = sample_X.shape[1]
num_classes = len(le.classes_)
model = FCNClassifier(input_channels, time_steps, num_classes).to(device)
model.load_state_dict(torch.load(saved_model_path))
model.eval()
logger.info("Model loaded and set to evaluation mode.")


# Evaluate model
logger.info("Starting model evaluation...")
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)


accuracy = 100 * correct / total
logger.info(f"Model evaluation completed. Test Accuracy: {accuracy:.2f}%")
print(f"Test Accuracy: {accuracy:.2f}%")
