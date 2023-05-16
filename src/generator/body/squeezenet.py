import torch
from torch import nn, Tensor

__all__ = ['SqueezeNet']

class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Sequential(
                nn.Conv2d(inplanes, squeeze_planes, kernel_size=1),
                nn.ReLU(inplace=True),
        )
        self.expand1x1 = nn.Sequential(
                nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1),
                nn.ReLU(inplace=True),
        )
        self.expand3x3 = nn.Sequential(
                nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.squeeze(x)
        return torch.cat([self.expand1x1(x), self.expand3x3(x)], dim=1)

class SqueezeNet(nn.Module):
    def __init__(self,
                 squeeze_planes: int = 8,
                 expand1x1_planes: int = 32,
                 expand3x3_planes: int = 32,
                 num_blocks: int = 16,
        ) :
        super().__init__()
        body = nn.Sequential(
            *[Fire(64, squeeze_planes, expand1x1_planes, expand3x3_planes) for _ in range(num_blocks)]
        )
        self.body = body

    def forward(self, x) :
        return self.body(x)
