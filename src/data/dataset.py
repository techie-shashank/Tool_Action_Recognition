import torch
from torch.utils.data import Dataset

class ToolTrackingWindowDataset(Dataset):
    def __init__(self, Xt, y):
        self.X = torch.tensor(Xt, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
