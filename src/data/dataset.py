import torch
from torch.utils.data import Dataset

from data.augmentation import augment


class ToolTrackingWindowDataset(Dataset):
    def __init__(self, Xt, y, augment=False, minority_classes=None):
        self.X = torch.tensor(Xt, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        self.minority_classes = minority_classes if minority_classes is not None else set()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        label = self.y[idx]

        if self.augment and label in self.minority_classes:
            x = augment(x)

        return x, label
