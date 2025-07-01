import torch
import torch.nn as nn
import torch.nn.functional as F

from data.augmentation import augment_batch
from logger import logger
from utils import train_model

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


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        return self.net(x)


def contrastive_pretrain(model, unlabeled_loader, device, epochs=10):
    projection_head = ProjectionHead(model.encoder_output_size).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(projection_head.parameters()),
        lr=1e-3
    )

    model.train()
    projection_head.train()
    logger.info("Starting contrastive pretraining...")

    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, _ in unlabeled_loader:
            x = x_batch.to(device)

            x1 = augment_batch(x_batch, device=device)
            x2 = augment_batch(x_batch, device=device)

            optimizer.zero_grad()
            h1 = model.forward_encoder_only(x1)
            h2 = model.forward_encoder_only(x2)

            # Pass through projection head
            z1 = projection_head(h1)
            z2 = projection_head(h2)

            # Normalize embeddings (recommended)
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

            loss = nt_xent_loss(z1, z2, temperature= config["semi_supervised"].get("temperature"))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(unlabeled_loader)
        logger.info(f"[Contrastive Epoch {epoch + 1}] Pretraining Loss: {avg_loss:.4f}")


def train_contrastive(model, labeled_loader, unlabeled_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    logger.info("Using contrastive learning strategy for semi-supervised learning.")
    contrastive_pretrain(model, unlabeled_loader, device, epochs=10)

    logger.info("Fine-tuning on labeled data...")
    train_model(model, labeled_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)
