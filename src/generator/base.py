import torch
import math
from torch import nn

class UpsampleBlock(nn.Sequential):
    def __init__(self, channels):
        conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        pixel_shuffle = nn.PixelShuffle(2)
        prelu = nn.PReLU()
        super(UpsampleBlock, self).__init__(conv, pixel_shuffle, prelu)

class SRBaseNet(nn.Module):
    def __init__(self,
                 body: nn.Module,
                 head: nn.Module,
                 scale_factor = 4
        ) :
        super().__init__()
        upsample_block_num = int(math.log(scale_factor, 2))
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.body = body
        self.last_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.upsample_blocks = nn.Sequential(
            *[UpsampleBlock(64) for _ in range(upsample_block_num)],
        )
        self.head = head

    def forward(self, x):
        x = self.in_conv(x)
        x = x + self.last_block(self.body(x))
        x = self.upsample_blocks(x)
        return self.head(x)

    @torch.no_grad()
    def inference(self, lr) :
        sr = self.forward(lr)
        torch.clamp_(sr, 0, 1)
        return sr

    @torch.no_grad()
    def initialize_weights(self, pretrained_model = None) :
        if pretrained_model is not None :
            self.load_state_dict(torch.load(pretrained_model, map_location=self.device))
        else :
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    m.weight.data *= 0.1
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    m.weight.data *= 0.1
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def save(self, save_path) :
        torch.save(self.state_dict(), save_path)
