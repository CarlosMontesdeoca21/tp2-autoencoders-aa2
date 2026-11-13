import json
import math
import os

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from autoencoder import Autoencoder
from data_loader import data_loader_valid
from noise import add_gaussian_noise

NOISE_STD = 0.35
CHECKPOINT_PATH = os.path.join('checkpoints', 'noise_autoencoder.pt')
RESULTS_DIR = 'results'
NUM_IMAGES = 10

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = Autoencoder().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dataiter = iter(data_loader_valid)
images, _ = next(dataiter)
images = images[:NUM_IMAGES].to(device)

with torch.no_grad():
    noisy = add_gaussian_noise(images, std=NOISE_STD)
    denoised = model(noisy)


def mse(a, b):
    return torch.mean((a - b) ** 2).item()


def psnr(a, b):
    return 10.0 * math.log10(1.0 / mse(a, b))


metrics = {
    'noise_std': NOISE_STD,
    'mse_real_vs_noisy': mse(images, noisy),
    'mse_real_vs_denoised': mse(images, denoised),
    'psnr_real_vs_noisy': psnr(images, noisy),
    'psnr_real_vs_denoised': psnr(images, denoised),
}

metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
with open(metrics_path) as f:
    full_metrics = json.load(f)
full_metrics['test'] = metrics
with open(metrics_path, 'w') as f:
    json.dump(full_metrics, f, indent=2)

print(f"MSE  real vs noisy:    {metrics['mse_real_vs_noisy']:.4f}")
print(f"MSE  real vs denoised: {metrics['mse_real_vs_denoised']:.4f}")
print(f"PSNR real vs noisy:    {metrics['psnr_real_vs_noisy']:.2f} dB")
print(f"PSNR real vs denoised: {metrics['psnr_real_vs_denoised']:.2f} dB")

real = images.cpu()
noisy = noisy.cpu()
denoised = denoised.cpu()

plt.figure(figsize=(2 * NUM_IMAGES, 6.5))
plt.gray()

for i in range(NUM_IMAGES):
    plt.subplot(3, NUM_IMAGES, i + 1)
    plt.imshow(real[i, 0])
    if i == 0:
        plt.ylabel('Real', fontsize=12)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, NUM_IMAGES, NUM_IMAGES + i + 1)
    plt.imshow(noisy[i, 0])
    if i == 0:
        plt.ylabel(f'Noisy\n(std={NOISE_STD})', fontsize=12)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, NUM_IMAGES, 2 * NUM_IMAGES + i + 1)
    plt.imshow(denoised[i, 0])
    if i == 0:
        plt.ylabel('Denoised', fontsize=12)
    plt.xticks([])
    plt.yticks([])

plt.suptitle('Autoencoder - Noise cancellation', fontsize=14)
plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, 'noise_cancellation.png')
plt.savefig(out_path, dpi=150)
print(f'Results saved to {out_path}')
