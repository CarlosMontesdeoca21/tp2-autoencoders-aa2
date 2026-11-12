import torch


def add_gaussian_noise(images, std=0.35):
    noisy = images + torch.randn_like(images) * std
    return torch.clamp(noisy, 0.0, 1.0)
