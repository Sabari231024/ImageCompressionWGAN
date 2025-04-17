import torch
import torch.nn as nn
from linformer import Linformer

class Encoder(nn.Module):
    def __init__(self, num_channels_in_encoder=8):  
        super(Encoder, self).__init__()
        self.e_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),  
            nn.ReLU()
        )
        self.e_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),  
            nn.ReLU()
        )
        self.e_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.linformer_block = None
        self.e_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1), 
            nn.ReLU()
        )
        self.gumbel_quant = GumbelQuantize(num_hiddens=64, n_embed=64 ,embedding_dim=8)
    def forward(self, x):
        ec1 = self.e_conv_1(x)  
        ec2 = self.e_conv_2(ec1)  
        eblock1 = self.e_block_1(ec2) + ec2 
        batch_size, channels, height, width = eblock1.shape
        seq_len = height * width  
        if self.linformer_block is None:
            self.linformer_block = Linformer(
                dim=64, seq_len=seq_len, depth=1, heads=4, k=64
            ).to(x.device)  
        eblock1_flat = eblock1.view(batch_size, seq_len, channels)  
        linform = self.linformer_block(eblock1_flat) 
        linform_reshaped = linform.view(batch_size, channels, height, width)
        ec3 = self.e_conv_3(linform_reshaped)  
        z_q, diff = self.gumbel_quant(ec3)
        return z_q, diff
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netE = Encoder(num_channels_in_encoder=8).to(device)
inp = torch.randn(IMG_WIDTH * IMG_HEIGHT * 3 * 100).view((-1, 3, IMG_HEIGHT, IMG_WIDTH)).to(device)
output = netE(inp)[0]
print(output.shape)  
print('The Compression Ratio is :  ' + str((output.shape[1] * output.shape[2] * output.shape[3]) / (IMG_WIDTH * IMG_HEIGHT * 3) * 100))
