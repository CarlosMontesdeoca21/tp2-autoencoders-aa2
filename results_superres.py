import json
import math
import os

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from superresolution import Autoencoder_SuperRes, downsample
from data_loader import data_loader_valid

CHECKPOINT_PATH = os.path.join('checkpoints', 'superres_autoencoder.pt')
RESULTS_DIR = 'results'
NUM_PER_GROUP = 5
COLS = 2 * NUM_PER_GROUP        
FACTOR_14 = 2                   
FACTOR_7 = 4                    

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = Autoencoder_SuperRes().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dataiter = iter(data_loader_valid)
images, _ = next(dataiter)
images = images[:COLS].to(device)          

with torch.no_grad():
    small_14 = downsample(images[:NUM_PER_GROUP], FACTOR_14)   
    small_7 = downsample(images[NUM_PER_GROUP:], FACTOR_7)     
    super_14 = model(small_14)
    super_7 = model(small_7)

up_14 = F.interpolate(small_14, size=(28, 28), mode='nearest')
up_7 = F.interpolate(small_7, size=(28, 28), mode='nearest')


def mse(a, b):
    return torch.mean((a - b) ** 2).item()


def psnr(a, b):
    return 10.0 * math.log10(1.0 / mse(a, b))


metrics = {
    '14x14': {
        'mse_real_vs_upsampled': mse(images[:NUM_PER_GROUP], up_14),
        'mse_real_vs_superres': mse(images[:NUM_PER_GROUP], super_14),
        'psnr_real_vs_upsampled': psnr(images[:NUM_PER_GROUP], up_14),
        'psnr_real_vs_superres': psnr(images[:NUM_PER_GROUP], super_14),
    },
    '7x7': {
        'mse_real_vs_upsampled': mse(images[NUM_PER_GROUP:], up_7),
        'mse_real_vs_superres': mse(images[NUM_PER_GROUP:], super_7),
        'psnr_real_vs_upsampled': psnr(images[NUM_PER_GROUP:], up_7),
        'psnr_real_vs_superres': psnr(images[NUM_PER_GROUP:], super_7),
    },
}

metrics_path = os.path.join(RESULTS_DIR, 'metrics_superres.json')
with open(metrics_path) as f:
    full_metrics = json.load(f)
full_metrics['test'] = metrics
with open(metrics_path, 'w') as f:
    json.dump(full_metrics, f, indent=2)

for name in ('14x14', '7x7'):
    m = metrics[name]
    print(f'[{name}] MSE  baseline {m["mse_real_vs_upsampled"]:.4f} -> superres {m["mse_real_vs_superres"]:.4f}')
    print(f'[{name}] PSNR baseline {m["psnr_real_vs_upsampled"]:.2f} dB -> superres {m["psnr_real_vs_superres"]:.2f} dB')

real = images.cpu()

plt.figure(figsize=(2 * COLS, 6.5))
plt.gray()

for i in range(COLS):
    if i < NUM_PER_GROUP:
        small = small_14[i].cpu()
        out = super_14[i].cpu()
    else:
        small = small_7[i - NUM_PER_GROUP].cpu()
        out = super_7[i - NUM_PER_GROUP].cpu()

    plt.subplot(3, COLS, i + 1)
    plt.imshow(real[i, 0])
    if i == 0:
        plt.title('14x14 -> 28x28')
        plt.ylabel('Real\n(28x28)')
    elif i == NUM_PER_GROUP:
        plt.title('7x7 -> 28x28')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, COLS, COLS + i + 1)
    plt.imshow(small[0])
    if i == 0:
        plt.ylabel('Reducida\n(14x14 / 7x7)')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, COLS, 2 * COLS + i + 1)
    plt.imshow(out[0])
    if i == 0:
        plt.ylabel('Super-resolución\n(28x28)')
    plt.xticks([])
    plt.yticks([])

plt.suptitle('Autoencoder - Super-resolution', fontsize=14)
plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, 'superresolution.png')
plt.savefig(out_path, dpi=150)
print(f'Results saved to {out_path}')