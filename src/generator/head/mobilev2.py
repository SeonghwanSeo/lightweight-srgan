import torch
from torch import nn, Tensor

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
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int) :
        super().__init__()
        if expand_ratio != 1:
            hidden_dim = in_channels * expand_ratio
            self.layers = nn.Sequential(
                BasicConv(in_channels, hidden_dim, kernel_size=1),
                BasicConv(hidden_dim, hidden_dim, kernel_size=9, groups=hidden_dim),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            )
        else :
            self.layers = nn.Sequential(
                BasicConv(in_channels, in_channels, kernel_size=9, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

class MobileHeadV2(nn.Module):
    def __init__(self,
                 expand_ratio: int = 1,
        ) :
        super().__init__()
        self.layer = InvertedResidual(64, 3, expand_ratio)

    def forward(self, x) :
        return self.layer(x)
