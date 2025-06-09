import torch
import torch.nn.functional as F

from src.data.augmentation import augment
from src.logger import logger
from src.utils import train_model

import os
import json

config_path = os.path.join(r'../', "config.json")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)

    similarity_matrix = torch.matmul(representations, representations.T)
    labels = torch.arange(batch_size).to(z1.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

    positives = torch.sum(z1 * z2, dim=-1)
    positives = torch.cat([positives, positives], dim=0)

    loss = -torch.log(
        torch.exp(positives / temperature) /
        torch.sum(torch.exp(similarity_matrix / temperature), dim=1)
    )
    return loss.mean()

def contrastive_pretrain(model, unlabeled_loader, optimizer, device, epochs=20):
    model.train()
    logger.info("Starting contrastive pretraining...")

    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch in unlabeled_loader:
            x = x_batch[0].to(device)  # Unpack (X,) if needed

            x1 = augment(x)
            x2 = augment(x)

            optimizer.zero_grad()
            z1 = model(x1)  # Assume model returns embeddings (not logits)
            z2 = model(x2)

            loss = nt_xent_loss(z1, z2, temperature= config["semi_supervised"].get("temperature"))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(unlabeled_loader)
        logger.info(f"[Contrastive Epoch {epoch+1}] Pretraining Loss: {avg_loss:.4f}")


def train_contrastive(model, labeled_loader, unlabeled_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    logger.info("Using contrastive learning strategy for semi-supervised learning.")
    contrastive_pretrain(model, unlabeled_loader, optimizer, device, epochs=num_epochs)

    logger.info("Fine-tuning on labeled data...")
    train_model(model, labeled_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)
