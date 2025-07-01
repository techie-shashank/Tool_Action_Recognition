import numpy as np
from torch.utils.data import random_split, DataLoader, ConcatDataset
from logger import logger
from semi_supervised.contrastive import train_contrastive
from semi_supervised.pseudo_labeling import train_pseudo_labelling
from utils import config, train_model, get_weighted_sampler


def split_training_dataset(train_dataset, labeled_ratio=0.1):
    """
    Splits the training dataset into labeled and unlabeled subsets.
    """
    total_size = len(train_dataset)
    labeled_size = int(labeled_ratio * total_size)
    unlabeled_size = total_size - labeled_size

    logger.info(f"Splitting dataset: {labeled_size} labeled, {unlabeled_size} unlabeled.")

    return random_split(train_dataset, [labeled_size, unlabeled_size])

def train_semi_supervised(model, train_dataset, val_dataset, criterion, optimizer, device, num_epochs=10):
    batch_size = config["batch_size"]
    labeled_dataset, unlabeled_dataset = split_training_dataset(
        train_dataset, labeled_ratio=config["semi_supervised"]["labelled_ratio"]
    )
    data_balancing = config.get('data_balancing', [])
    labels = np.array([labeled_dataset.dataset.y[idx].item() for idx in labeled_dataset.indices])
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, sampler=get_weighted_sampler(
        labels
    )) if "weighted_sampling" in data_balancing else DataLoader(
        labeled_dataset, batch_size=batch_size, shuffle=True
    )
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    if config["semi_supervised"]["strategy"] == "pseudo_labeling":
        logger.info("Using pseudo-labeling strategy for semi-supervised learning.")
        train_pseudo_labelling(model, labeled_loader, unlabeled_loader, val_loader, criterion, optimizer, device, num_epochs)
    elif config["semi_supervised"]["strategy"] == "contrastive":
        logger.info("Using contrastive learning strategy for semi-supervised learning.")
        train_contrastive(model, labeled_loader, unlabeled_loader, val_loader, criterion, optimizer, device, num_epochs)
