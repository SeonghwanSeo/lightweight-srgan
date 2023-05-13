import torch
from torch import nn

class BasicBlock(nn.Sequential) :
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) :
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

class VGGStyleDiscriminator(nn.Module):
    def __init__(self):
        super(VGGStyleDiscriminator, self).__init__()
        self.input_size = 96
        self.inc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            BasicBlock(64, 64, kernel_size=3, stride=2, padding=1),     # /2
            BasicBlock(64, 128, kernel_size=3, stride=1, padding=1),
            BasicBlock(128, 128, kernel_size=3, stride=2, padding=1),   # /4
            BasicBlock(128, 256, kernel_size=3, stride=1, padding=1),
            BasicBlock(256, 256, kernel_size=3, stride=2, padding=1),   # /8
            BasicBlock(256, 512, kernel_size=3, stride=1, padding=1),
            BasicBlock(512, 512, kernel_size=3, stride=2, padding=1),   # /16
        )
        after_block_size = self.input_size // 16
        self.head = nn.Sequential(
            nn.Linear(512 * after_block_size ** 2, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            #nn.Sigmoid(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        x = torch.clamp_(x, 0, 1)
        x = self.inc(x)     # B, 3, 96, 96  -> B, 64, 96, 96
        x = self.blocks(x)  # B, 64, 96, 96 -> B, 512, 6, 6
        x = x.view(B, -1)   # B, 512, 6, 6  -> B, 512*6*6
        x = self.head(x)    # B, 512*6*6    -> B, 1
        return x.squeeze(0) # B, 1          -> B,
    
    def initialize_weights(self) :
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

