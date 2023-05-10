import torch
from torch import nn, Tensor

from .base import SRBaseNet

class BasicConv(nn.Sequential) :
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=None,
                 dilation=1,
                 groups=1
        ) :
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

class InvertedResidual(nn.Module):
    def __init__(self, channels: int, expand_ratio: int) :
        super().__init__()
        if expand_ratio != 1:
            hidden_dim = channels * expand_ratio
            self.layers = nn.Sequential(
                BasicConv(channels, hidden_dim, kernel_size=1),
                BasicConv(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim),
                nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
        else :
            self.layers = nn.Sequential(
                BasicConv(channels, channels, kernel_size=3, groups=channels),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(x)

class SRMobileNetV2(SRBaseNet):
    def __init__(self,
                 hidden_dim: int = 32,
                 expand_ratio: int = 4,
                 num_blocks = 5,
                 scale_factor = 4) :
        if hidden_dim != 64 :
            body = nn.Sequential(
                BasicConv(64, hidden_dim, kernel_size=1),
                *[InvertedResidual(hidden_dim, expand_ratio) for _ in range(num_blocks)],
                BasicConv(hidden_dim, 64, kernel_size=1),
            )
        else :
            body = nn.Sequential(
                *[InvertedResidual(64, expand_ratio) for _ in range(num_blocks)],
            )
        super().__init__(body, scale_factor)
