import torch
import numpy as np
from scipy.signal import resample

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def to_tensor(x, device='cpu'):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    return x

def random_jitter(x, sigma=0.05):
    if isinstance(x, torch.Tensor):
        return x + sigma * torch.randn_like(x)
    else:
        return x + np.random.normal(0, sigma, x.shape)


def random_scaling(x, sigma=0.1):
    if isinstance(x, torch.Tensor):
        # Generate one scale factor per feature (channel), shape: (1, features)
        factor = torch.normal(mean=1.0, std=sigma, size=(1, x.size(1)), device=x.device)
        return x * factor
    else:
        # Numpy version for non-torch inputs
        factor = np.random.normal(1.0, sigma, size=(1, x.shape[1]))
        return x * factor

def add_gaussian_noise(x, noise_level=0.01):
    if isinstance(x, torch.Tensor):
        noise = torch.randn_like(x) * noise_level
        return x + noise
    else:
        noise = np.random.normal(0, noise_level, x.shape)
        return x + noise

def magnitude_scaling(x, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return x * scale

def time_warp(x, warp_factor_range=(0.8, 1.2)):
    x_np = to_numpy(x)
    factor = np.random.uniform(*warp_factor_range)
    n_steps = max(1, int(x_np.shape[0] * factor))

    x_warped = resample(x_np, n_steps, axis=0)

    # Ensure original number of dimensions
    if n_steps > x_np.shape[0]:
        x_warped = x_warped[:x_np.shape[0]]
    else:
        pad_width = x_np.shape[0] - n_steps
        if x_warped.ndim == 1:
            # For 1D case
            x_warped = np.pad(x_warped, (0, pad_width), mode='edge')
        elif x_warped.ndim == 2:
            # For 2D (time x features)
            x_warped = np.pad(x_warped, ((0, pad_width), (0, 0)), mode='edge')
        else:
            raise ValueError(f"Unexpected shape for x_warped: {x_warped.shape}")

    if isinstance(x, torch.Tensor):
        return to_tensor(x_warped, device=x.device)
    return x_warped


def augment(x, augmentations=None, device='cpu'):
    if augmentations is None:
        augmentations = ["jitter", "scaling", "gaussian_noise", "magnitude_scaling", "time_warp"]

    x_aug = x

    if "jitter" in augmentations and np.random.rand() < 0.5:
        x_aug = random_jitter(x_aug)
    if "scaling" in augmentations and np.random.rand() < 0.5:
        x_aug = random_scaling(x_aug)
    if "gaussian_noise" in augmentations and np.random.rand() < 0.3:
        x_aug = add_gaussian_noise(x_aug)
    if "magnitude_scaling" in augmentations and np.random.rand() < 0.3:
        x_aug = magnitude_scaling(x_aug)
    if "time_warp" in augmentations and np.random.rand() < 0.2:
        x_aug = time_warp(x_aug)

    # Convert to original type
    if isinstance(x, np.ndarray) and isinstance(x_aug, torch.Tensor):
        x_aug = to_numpy(x_aug)
    if isinstance(x, torch.Tensor) and isinstance(x_aug, np.ndarray):
        x_aug = to_tensor(x_aug, device=device)

    return x_aug


def augment_batch(x_batch, augmentations=None, device='cpu'):
    augmented_samples = [
        augment(xi, augmentations=augmentations, device=device)
        for xi in x_batch
    ]
    return torch.stack(augmented_samples).to(device)
