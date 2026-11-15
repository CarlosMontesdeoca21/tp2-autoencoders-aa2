import json
import os
import random

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from superresolution import Autoencoder_SuperRes, downsample
from data_loader import data_loader, data_loader_valid

NUM_EPOCHS = 40        # techo de épocas: el early stopping puede parar antes
PATIENCE = 5           # épocas seguidas sin mejorar en validación antes de parar
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
val_losses = []
best_val = float('inf')     # la mejor validación hasta ahora; inf -> la 1ª época siempre mejora
best_epoch = 0
epochs_without_improvement = 0

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

    # --- Validación: medimos SOBRE EL TEST (imágenes nunca vistas) ---
    model.eval()
    val_running = 0.0
    val_batches = 0
    with torch.no_grad():                       # solo medir, no aprender
        for img, _ in data_loader_valid:
            img = img.to(device)
            for factor in FACTORS:              # probamos LOS DOS tamaños, siempre igual
                small = downsample(img, factor)
                recon = model(small)
                val_running += criterion(recon, img).item()
                val_batches += 1
    avg_val = val_running / val_batches
    val_losses.append(avg_val)

    print(f'Epoch:{epoch + 1}, Loss:{avg_loss:.4f}, Val:{avg_val:.4f}')

    # --- Early stopping: guarda el MEJOR modelo y para si deja de mejorar ---
    if avg_val < best_val:
        best_val = avg_val
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'factors': FACTORS,
            'epoch': best_epoch,
            'val_loss': best_val,
            'epoch_losses': epoch_losses,
            'val_losses': val_losses,
        }, CHECKPOINT_PATH)
        print(f'   -> mejor modelo hasta ahora (val {avg_val:.4f})')
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f'Early stopping en la época {epoch + 1} (mejor modelo: época {best_epoch})')
            break

with open(os.path.join(RESULTS_DIR, 'metrics_superres.json'), 'w') as f:
    json.dump({
        'factors': list(FACTORS),
        'num_epochs': NUM_EPOCHS,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'epoch_losses': epoch_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val,
        'epochs_run': len(epoch_losses),
    }, f, indent=2)

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label='Train')
plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', label='Test')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs test loss (super-resolution)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'loss_curve_superres.png'))
print(f'Checkpoint saved to {CHECKPOINT_PATH}')
print(f'Loss curve saved to {os.path.join(RESULTS_DIR, "loss_curve_superres.png")}')
