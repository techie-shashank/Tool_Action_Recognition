import torch

def random_jitter(x, sigma=0.05):
    return x + sigma * torch.randn_like(x)

def random_scaling(x, sigma=0.1):
    factor = torch.normal(mean=1.0, std=sigma, size=(x.size(0), 1, 1), device=x.device)
    return x * factor

def augment(x):
    if torch.rand(1).item() > 0.5:
        return random_jitter(x)
    else:
        return random_scaling(x)
