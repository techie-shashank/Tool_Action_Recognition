import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import logger
from utils import train_model  # your existing supervised training function

import os
import json

config_path = os.path.join(r'../', "config.json")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)


def update_ema_variables(student_model, teacher_model, ema_decay):
    """
    Update teacher parameters as EMA of student parameters.
    teacher = ema_decay * teacher + (1 - ema_decay) * student
    """
    for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_param.data.mul_(ema_decay).add_(student_param.data * (1 - ema_decay))


def get_consistency_loss(student_logits, teacher_logits, loss_type="mse"):
    """
    Calculate consistency loss between student and teacher outputs.
    Supported loss types: mse, kl, ce
    """
    if loss_type == "mse":
        # MSE on softmax probabilities
        student_prob = F.softmax(student_logits, dim=1)
        teacher_prob = F.softmax(teacher_logits, dim=1)
        loss = F.mse_loss(student_prob, teacher_prob)
    elif loss_type == "kl":
        # KL divergence between teacher and student probs
        student_log_prob = F.log_softmax(student_logits, dim=1)
        teacher_prob = F.softmax(teacher_logits, dim=1)
        loss = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean')
    elif loss_type == "ce":
        # Cross entropy using teacher predictions as soft targets (soft labels)
        teacher_prob = F.softmax(teacher_logits, dim=1)
        loss = -(teacher_prob * F.log_softmax(student_logits, dim=1)).sum(dim=1).mean()
    else:
        raise ValueError(f"Unsupported consistency loss type: {loss_type}")
    return loss


def train_mean_teacher(model, labeled_loader, unlabeled_loader, val_loader, criterion, optimizer, device,
                       consistency_weight=1.0, ema_decay=0.99, consistency_type="mse", num_epochs=10):
    import copy
    import matplotlib.pyplot as plt

    teacher_model = copy.deepcopy(model)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Store losses for plotting
    epoch_total_losses = []
    epoch_supervised_losses = []
    epoch_consistency_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        supervised_loss_total = 0.0
        consistency_loss_total = 0.0

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        num_batches = max(len(labeled_loader), len(unlabeled_loader))

        for _ in range(num_batches):
            optimizer.zero_grad()

            try:
                x_labeled, y_labeled = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_labeled, y_labeled = next(labeled_iter)
            x_labeled, y_labeled = x_labeled.to(device), y_labeled.to(device)

            try:
                x_unlabeled, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                x_unlabeled, _ = next(unlabeled_iter)
            x_unlabeled = x_unlabeled.to(device)

            outputs_labeled = model(x_labeled)
            loss_supervised = criterion(outputs_labeled, y_labeled)

            outputs_unlabeled_student = model(x_unlabeled)
            with torch.no_grad():
                outputs_unlabeled_teacher = teacher_model(x_unlabeled)

            loss_consistency = get_consistency_loss(outputs_unlabeled_student, outputs_unlabeled_teacher, consistency_type)

            loss = loss_supervised + consistency_weight * loss_consistency
            loss.backward()
            optimizer.step()

            update_ema_variables(model, teacher_model, ema_decay)

            total_loss += loss.item()
            supervised_loss_total += loss_supervised.item()
            consistency_loss_total += loss_consistency.item()

        # Save losses for the epoch
        avg_loss = total_loss / num_batches
        avg_supervised = supervised_loss_total / num_batches
        avg_consistency = consistency_loss_total / num_batches

        epoch_total_losses.append(avg_loss)
        epoch_supervised_losses.append(avg_supervised)
        epoch_consistency_losses.append(avg_consistency)

        logger.info(
            f"[Epoch {epoch + 1}] Total Loss: {avg_loss:.4f} "
            f"(Supervised: {avg_supervised:.4f}, Consistency: {avg_consistency:.4f})"
        )

    logger.info("Mean Teacher training finished.")

    # Plot the learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_total_losses, label="Total Loss")
    plt.plot(epoch_supervised_losses, label="Supervised Loss")
    plt.plot(epoch_consistency_losses, label="Consistency Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Mean Teacher Training Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
