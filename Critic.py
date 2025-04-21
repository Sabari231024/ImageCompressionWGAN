import torch
import torch.nn as nn
import torch.nn.utils as utils

class WGAN_Discriminator(nn.Module):  # critic
    def __init__(self):
        super(WGAN_Discriminator, self).__init__()
        self.latent_layer1 = nn.Sequential(
            utils.spectral_norm(nn.ConvTranspose2d(8, 16, (3,3), stride=2, padding=1, output_padding=1)),
            nn.ReLU(inplace=True),
        )
        self.latent_layer2 = nn.Sequential(
            utils.spectral_norm(nn.ConvTranspose2d(16, 32, (3,3), stride=2, padding=1, output_padding=1)),
            nn.ReLU(inplace=True),
        )
        self.latent_layer3 = nn.Sequential(
            utils.spectral_norm(nn.ConvTranspose2d(32, 64, (3,3), stride=2, padding=1, output_padding=1)),
            nn.ReLU(inplace=True),
            utils.spectral_norm(nn.ConvTranspose2d(64, 64, (3,3), stride=1, padding=1, output_padding=0)),
            nn.ReLU(inplace=True),
        )
        self.latent_layer4 = nn.Sequential(
            utils.spectral_norm(nn.ConvTranspose2d(64, 3, (3,3), stride=1, padding=1, output_padding=0)),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(6, 64, 3, 1, 1)),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(64, 64, 4, 2, 1)),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(64, 32, 4, 2, 1)),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(32, 16, 4, 2, 1)),
            nn.ReLU(inplace=True),
            utils.spectral_norm(nn.Conv2d(16, 2, 3, 1, 1)),
            nn.ReLU(inplace=True),
        )

        self.fc1 = utils.spectral_norm(nn.Linear(2 * 28 * 28, 256))
        self.fc2 = utils.spectral_norm(nn.Linear(256, 100))
        self.fc3 = utils.spectral_norm(nn.Linear(100, 1))

    def forward(self, inp):
        encoded = inp['encoded'].to(device)
        x = inp['img'].to(device)
        y = self.latent_layer1(encoded)
        y = self.latent_layer2(y)
        y = self.latent_layer3(y)
        y = self.latent_layer4(y)
        x = torch.cat((x, y), 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
