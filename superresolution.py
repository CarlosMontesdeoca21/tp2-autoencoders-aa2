import torch.nn as nn
import torch.nn.functional as F


def downsample(images, factor):
    return F.avg_pool2d(images, factor)


class Autoencoder_SuperRes(nn.Module):

    def __init__(self):
        super().__init__()
    
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),   
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),  
            nn.ReLU(),
        )

        
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),  
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.interpolate(x, size=(28, 28), mode='bilinear')  
        encoded = self.encoder(x)                             
        decoded = self.decoder(encoded)                       
        return decoded