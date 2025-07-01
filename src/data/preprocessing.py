from collections import Counter
import math

import torch
import numpy as np
from logger import logger
from data.augmentation import augment
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp


def preprocess_signals(X_train, X_validation, X_test):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))
    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)

def split_data(X, y, train_ratio=0.7, val_ratio=0.15, seed=42, data_ratio=1.0):
    torch.manual_seed(seed)
    total_size = len(X)

    # Generate shuffled indices
    indices = torch.randperm(total_size).tolist()

    # Compute split sizes
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Split data
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]

    X_val = [X[i] for i in val_indices]
    y_val = [y[i] for i in val_indices]

    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]

    logger.info(f"Dataset split into Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Optionally reduce training data size (data_ratio control)
    data_ratio = data_ratio
    if data_ratio < 1.0:
        reduced_size = int(data_ratio * len(X_train))
        X_train = X_train[:reduced_size]
        y_train = y_train[:reduced_size]
        logger.info(f"Reduced training data to {len(X_train)} samples due to data_ratio={data_ratio}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def resample_dataset(X, y, target_ratio=0.3, random_state=42, apply_augmentation=True):
    np.random.seed(random_state)

    y_arr = np.array(y)  # for easy indexing on y labels
    label_counts = Counter(y_arr)
    dominant_class = label_counts.most_common(1)[0][0]
    dominant_count = label_counts[dominant_class]

    X_resampled = []
    y_resampled = []

    for label in label_counts:
        # get indices of samples for this label
        label_indices = [i for i, lbl in enumerate(y) if lbl == label]
        count = label_counts[label]

        if label == dominant_class:
            n_samples = int(dominant_count * target_ratio)
            chosen_indices = np.random.choice(label_indices, n_samples, replace=False)
        else:
            n_samples = max(count, int(dominant_count * target_ratio))
            chosen_indices = np.random.choice(label_indices, n_samples, replace=True)

        selected_samples = [X[i] for i in chosen_indices]

        if apply_augmentation and label != dominant_class:
            augmented_samples = [augment(sample) for sample in selected_samples]
            selected_samples = augmented_samples

        X_resampled.extend(selected_samples)
        y_resampled.extend([label] * len(selected_samples))

    # Shuffle the dataset (both X and y)
    perm = np.random.permutation(len(y_resampled))
    X_resampled = [X_resampled[i] for i in perm]
    y_resampled = [y_resampled[i] for i in perm]

    return X_resampled, y_resampled


def get_minority_classes(y, threshold=0.3):
    from collections import Counter
    counts = Counter(y)
    max_count = max(counts.values())
    minority_classes = {cls for cls, c in counts.items() if c < max_count * threshold}
    return minority_classes


def expand_dataset(X, y, minority_classes):
    augmented_X = []
    augmented_y = []

    class_counts = Counter(y)
    max_count = max(class_counts.values())
    target_count = 0.25 * max_count  # Minority classes should be at least 25% of max

    for i in range(len(X)):
        x_sample = X[i]
        label = y[i]

        # Add original sample as is
        augmented_X.append(x_sample)
        augmented_y.append(label)

        if label in minority_classes:
            current_count = class_counts[label]
            samples_needed = max(0, math.ceil(target_count - current_count))
            n_orig = current_count
            n_augments_per_sample = math.ceil(samples_needed / n_orig) if n_orig > 0 else 0

            for _ in range(n_augments_per_sample):
                x_aug = augment(x_sample)  # augment should accept and return same type as x_sample
                augmented_X.append(x_aug)
                augmented_y.append(label)

    return augmented_X, augmented_y


def balance_data(Xt, y, balancing_strategies):
    logger.info(
        f"Original data: samples={len(Xt)}, sample_shape={Xt[0].shape if len(Xt) > 0 else 'N/A'}, labels={len(y)}")
    if "oversample" in balancing_strategies:
        logger.info("Applying SMOTE oversampling...")

        # Convert list of ndarrays to single 2D array for SMOTE
        Xt_np = np.array(Xt)  # shape: (n_samples, time_steps, channels)
        Xt_2d = Xt_np.reshape(len(Xt_np), -1)  # shape: (n_samples, features)

        # SMOTE neighbor adjustment
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        k_neighbors = max(1, min(5, min_class_count - 1))

        if min_class_count < 2:
            logger.warning(f"SMOTE skipped: Too few samples in some classes (min_class_count={min_class_count})")
        else:
            smote = SMOTE(k_neighbors=k_neighbors)
            Xt_resampled, y_resampled = smote.fit_resample(Xt_2d, y)

            # Reshape back to (n_samples, time_steps, channels)
            Xt_resampled_3d = Xt_resampled.reshape(-1, Xt_np.shape[1], Xt_np.shape[2])

            # Convert back to list of ndarrays
            Xt = [sample for sample in Xt_resampled_3d]
            y = list(y_resampled)

            logger.info(f"After SMOTE: samples={len(Xt)}, labels={len(y)}, class_dist={Counter(y)}")

    if "resample" in balancing_strategies:
        logger.info("Applying balanced resampling (oversample minorities + undersample majority)...")
        Xt, y = resample_dataset(Xt, y, target_ratio=0.10)
        logger.info(
            f"After balanced resampling: samples={len(Xt)}, sample_shape={Xt[0].shape if len(Xt) > 0 else 'N/A'}, labels={len(y)}")
    if "augment" in balancing_strategies:
        logger.info("Applying data augmentation to minority classes...")
        minority_classes = get_minority_classes(y)
        Xt, y = expand_dataset(Xt, y, minority_classes)
        logger.info(
            f"After augmentation: samples={len(Xt)}, sample_shape={Xt[0].shape if len(Xt) > 0 else 'N/A'}, labels={len(y)}")
    return Xt, y