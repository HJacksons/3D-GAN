import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, input_dim=200):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(

            nn.ConvTranspose3d(input_dim, 512, kernel_size=4, stride=1, padding=0, bias=False, dtype=torch.float32),
            nn.BatchNorm3d(512),
            nn.ReLU(True),

            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1, bias=False, dtype=torch.float32),
            nn.BatchNorm3d(256),
            nn.ReLU(True),

            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=False, dtype=torch.float32),
            nn.BatchNorm3d(128),
            nn.ReLU(True),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, bias=False, dtype=torch.float32),
            nn.BatchNorm3d(64),
            nn.ReLU(True),

            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1, bias=False, dtype=torch.float32),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1, bias=False, dtype=torch.float32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1, bias=False, dtype=torch.float32),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1, bias=False, dtype=torch.float32),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1, bias=False, dtype=torch.float32),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=0, bias=False, dtype=torch.float32),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
