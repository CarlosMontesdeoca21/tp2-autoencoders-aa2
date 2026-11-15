import torch.nn as nn
import torch.nn.functional as F


def downsample(images, factor):
    return F.avg_pool2d(images, factor)


class Autoencoder_SuperRes(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)                     
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4),                                        
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),   
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded