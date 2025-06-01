import torch
import torch.nn as nn

# Semi-supervised learning model wrapper
class SemiSupervisedModel(nn.Module):
    def __init__(self, base_model):
        super(SemiSupervisedModel, self).__init__()
        self.model = base_model

    def forward(self, x):
        return self.model(x)

    def predict_pseudo_labels(self, x, threshold=0.95):
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            mask = max_probs > threshold
        return preds[mask], mask
