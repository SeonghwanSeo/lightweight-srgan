import torch
from torch import nn, Tensor

class Fire(nn.Module):
    def __init__(self,
                 inplanes: int,
                 squeeze_planes: int,
                 expand1x1_planes: int,
                 expand3x3_planes: int,
                 expand5x5_planes: int,
                 expand7x7_planes: int,
                 expand9x9_planes: int,
                 ) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Sequential(
                nn.Conv2d(inplanes, squeeze_planes, kernel_size=1),
                nn.ReLU(inplace=True),
        )
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand5x5 = nn.Conv2d(squeeze_planes, expand5x5_planes, kernel_size=5, padding=2)
        self.expand7x7 = nn.Conv2d(squeeze_planes, expand7x7_planes, kernel_size=7, padding=3)
        self.expand9x9 = nn.Conv2d(squeeze_planes, expand9x9_planes, kernel_size=9, padding=4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.squeeze(x)
        x = torch.cat([
            self.expand1x1(x),
            self.expand3x3(x),
            self.expand5x5(x),
            self.expand7x7(x),
            self.expand9x9(x),
        ], dim=1)
        return self.relu(x)

class SqueezeHead(nn.Module) :
    def __init__(self,
                 squeeze_planes = 8,
                 expand1x1_planes = 32,
                 expand3x3_planes = 16,
                 expand5x5_planes = 8,
                 expand7x7_planes = 4,
                 expand9x9_planes = 4,
        ) :
        super().__init__()
        self.fire = Fire(64, squeeze_planes,
                         expand1x1_planes, expand3x3_planes, expand5x5_planes,
                         expand7x7_planes, expand9x9_planes)
        hidden_dim = expand1x1_planes + expand3x3_planes + expand5x5_planes + expand7x7_planes + expand9x9_planes
        self.conv = nn.Conv2d(hidden_dim, 3, kernel_size = 1)

    def forward(self, x) :
        return self.conv(self.fire(x))

