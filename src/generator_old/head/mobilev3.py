from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Type

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
                 groups=1,
                 activation: Type[nn.Module] = nn.Hardswish
        ) :
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
            activation(inplace=True),
        )

class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        kernel_size: int,
        use_se: bool,
        use_hs: bool = False,
        se_layer: Callable[..., nn.Module] = partial(SqueezeExcitation, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if use_hs else nn.ReLU

        if expand_ratio != 1 :
            hidden_dim = in_channels * expand_ratio
            self.layers = nn.Sequential(
                BasicConv(in_channels, hidden_dim, kernel_size=1, activation=activation_layer),
                BasicConv(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, activation=activation_layer),
                se_layer(hidden_dim, hidden_dim // 4) if use_se else nn.Identity(),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            )
        else :
            self.layers = nn.Sequential(
                BasicConv(in_channels, in_channels, kernel_size, groups=in_channels, activation=activation_layer),
                se_layer(in_channels, in_channels // 4) if use_se else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

class MobileHeadV3(nn.Module):
    def __init__(self,
                 expand_ratio: int = 1,
                 kernel_size: int = 9,
                 use_se: bool = False,
        ) :
        super().__init__()
        self.layer = InvertedResidual(64, 3, expand_ratio, kernel_size, use_se)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)
