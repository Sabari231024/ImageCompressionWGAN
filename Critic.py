import torch
import torch.nn as nn

class WGAN_Discriminator(nn.Module): #critic
    def __init__(self):
        super(WGAN_Discriminator, self).__init__()
        self.latent_layer1 = nn.Sequential(
            nn.ConvTranspose2d(8, 16, (3,3), stride=2, padding=1, output_padding=1),  
            nn.ReLU(inplace=True),
        )
        self.latent_layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, (3,3), stride=2, padding=1, output_padding=1),  
            nn.ReLU(inplace=True),
        )
        self.latent_layer3 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, (3,3), stride=2, padding=1, output_padding=1),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, (3,3), stride=1, padding=1, output_padding=0), 
            nn.ReLU(inplace=True),
        )
        self.latent_layer4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, (3,3), stride=1, padding=1, output_padding=0),  
            nn.ReLU(inplace=True),
        )
    
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(2*28*28, 256)
        self.fc2 = nn.Linear(256, 100)  
        self.fc3 = nn.Linear(100,1)
    
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