import torch
from torch import nn, Tensor
from typing import List

class Fire(nn.Module):
    def __init__(self,
                 inplanes: int,
                 squeeze_planes: int,
                 expand_planes_list: List[int],
                 ) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Sequential(
                nn.Conv2d(inplanes, squeeze_planes, kernel_size=1),
                nn.ReLU(inplace=True),
        )
        expand1x1_planes, expand3x3_planes, expand5x5_planes, expand7x7_planes, expand9x9_planes = expand_planes_list
        if expand1x1_planes > 0 :
            self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        if expand3x3_planes > 0 :
            self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        if expand5x5_planes > 0 :
            self.expand5x5 = nn.Conv2d(squeeze_planes, expand5x5_planes, kernel_size=5, padding=2)
        if expand7x7_planes > 0 :
            self.expand7x7 = nn.Conv2d(squeeze_planes, expand7x7_planes, kernel_size=7, padding=3)
        if expand9x9_planes > 0 :
            self.expand9x9 = nn.Conv2d(squeeze_planes, expand9x9_planes, kernel_size=9, padding=4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.squeeze(x)
        xs = []
        if hasattr(self, 'expand1x1') :
            xs.append(self.expand1x1(x))
        if hasattr(self, 'expand3x3') :
            xs.append(self.expand3x3(x))
        if hasattr(self, 'expand5x5') :
            xs.append(self.expand5x5(x))
        if hasattr(self, 'expand7x7') :
            xs.append(self.expand7x7(x))
        if hasattr(self, 'expand9x9') :
            xs.append(self.expand9x9(x))
        
        x = torch.cat(xs, dim=1)
        return self.relu(x)

class SqueezeHead(nn.Module) :
    def __init__(self,
                 squeeze_planes = 8,
                 kernel_size: int = 9,
        ) :
        super().__init__()
        if kernel_size == 3 :
            expand_planes_list = [32, 32, 0, 0, 0]
        elif kernel_size == 5 :
            expand_planes_list = [32, 16, 16, 0, 0]
        elif kernel_size == 7 :
            expand_planes_list = [32, 16, 8, 8, 0]
        elif kernel_size == 9 :
            expand_planes_list = [32, 16, 8, 4, 4]
        else :
            raise NotImplemented

        self.fire = Fire(64, squeeze_planes, expand_planes_list)
        self.conv = nn.Conv2d(64, 3, kernel_size = 1)

    def forward(self, x) :
        return self.conv(self.fire(x))

