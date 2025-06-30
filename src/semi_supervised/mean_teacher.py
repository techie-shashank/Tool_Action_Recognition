import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import train_model
from logger import logger
import copy
import os
import json

# Load config
config_path = os.path.join(r'../', "config.json")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

def update_ema(student_model, teacher_model, alpha):
    for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)

def train_mean_teacher(student_model, labeled_loader, unlabeled_loader, val_loader,
                       criterion, optimizer, device, num_epochs=10, alpha=0.99, lambda_consistency=1.0):
    logger.info("Starting Mean Teacher training...")

    teacher_model = copy.deepcopy(student_model)
    teacher_model.eval()
    
    student_model.train()
    
    for epoch in range(num_epochs):
        student_model.train()
        teacher_model.eval()
        
        total_loss = 0.0
        
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        for step in range(min(len(labeled_loader), len(unlabeled_loader))):
            try:
                x_l, y_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_l, y_l = next(labeled_iter)

            try:
                x_u, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                x_u, _ = next(unlabeled_iter)

            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u = x_u.to(device)

            # Forward pass
            student_outputs_l = student_model(x_l)
            supervised_loss = criterion(student_outputs_l, y_l)

            # Consistency loss on unlabeled data
            with torch.no_grad():
                teacher_outputs_u = teacher_model(x_u)
            student_outputs_u = student_model(x_u)
            consistency_loss = F.mse_loss(student_outputs_u, teacher_outputs_u.detach())

            total_batch_loss = supervised_loss + lambda_consistency * consistency_loss

            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            # EMA update
            update_ema(student_model, teacher_model, alpha)

            total_loss += total_batch_loss.item()

        avg_loss = total_loss / len(labeled_loader)
        logger.info(f"[Epoch {epoch+1}] Total Loss: {avg_loss:.4f}")

        # Optional: Evaluate on validation set
        if val_loader:
            student_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    outputs = student_model(x_val)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == y_val).sum().item()
                    total += y_val.size(0)
            acc = 100 * correct / total
            logger.info(f"Validation Accuracy: {acc:.2f}%")
