import json
import os

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from autoencoder import Autoencoder
from data_loader import data_loader
from noise import add_gaussian_noise

NOISE_STD = 0.35
NUM_EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5

CHECKPOINT_PATH = os.path.join('checkpoints', 'noise_autoencoder.pt')
RESULTS_DIR = 'results'

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = Autoencoder().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

epoch_losses = []
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    num_batches = 0
    for img, _ in data_loader:
        img = img.to(device)
        noisy = add_gaussian_noise(img, std=NOISE_STD)

        recon = model(noisy)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches
    epoch_losses.append(avg_loss)
    print(f'Epoch:{epoch + 1}, Loss:{avg_loss:.4f}')

torch.save({
    'model_state_dict': model.state_dict(),
    'noise_std': NOISE_STD,
    'num_epochs': NUM_EPOCHS,
    'epoch_losses': epoch_losses,
}, CHECKPOINT_PATH)

with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
    json.dump({
        'noise_std': NOISE_STD,
        'num_epochs': NUM_EPOCHS,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'epoch_losses': epoch_losses,
    }, f, indent=2)

plt.figure(figsize=(8, 4))
plt.plot(range(1, NUM_EPOCHS + 1), epoch_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title(f'Training loss (noise std={NOISE_STD})')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'loss_curve.png'))
print(f'Checkpoint saved to {CHECKPOINT_PATH}')
print(f'Loss curve saved to {os.path.join(RESULTS_DIR, "loss_curve.png")}')
