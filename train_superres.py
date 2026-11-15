import json
import os
import random

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from superresolution import Autoencoder_SuperRes, downsample
from data_loader import data_loader

NUM_EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5
FACTORS = (2, 4)

CHECKPOINT_PATH = os.path.join('checkpoints', 'superres_autoencoder.pt')
RESULTS_DIR = 'results'

torch.manual_seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = Autoencoder_SuperRes().to(device)

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
        factor = random.choice(FACTORS)     
        small = downsample(img, factor)

        recon = model(small)
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
    'factors': FACTORS,
    'num_epochs': NUM_EPOCHS,
    'epoch_losses': epoch_losses,
}, CHECKPOINT_PATH)

with open(os.path.join(RESULTS_DIR, 'metrics_superres.json'), 'w') as f:
    json.dump({
        'factors': list(FACTORS),
        'num_epochs': NUM_EPOCHS,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'epoch_losses': epoch_losses,
    }, f, indent=2)

plt.figure(figsize=(8, 4))
plt.plot(range(1, NUM_EPOCHS + 1), epoch_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training loss (super-resolution 7x7/14x14 -> 28x28)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'loss_curve_superres.png'))
print(f'Checkpoint saved to {CHECKPOINT_PATH}')
print(f'Loss curve saved to {os.path.join(RESULTS_DIR, "loss_curve_superres.png")}')