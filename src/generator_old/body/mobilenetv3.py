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
        channels: int,
        expand_ratio: int,
        use_se: bool,
        use_hs: bool,
        se_layer: Callable[..., nn.Module] = partial(SqueezeExcitation, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if use_hs else nn.ReLU

        if expand_ratio != 1 :
            hidden_dim = channels * expand_ratio
            self.layers = nn.Sequential(
                BasicConv(channels, hidden_dim, kernel_size=1, activation=activation_layer),
                BasicConv(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim, activation=activation_layer),
                se_layer(hidden_dim, hidden_dim // 4) if use_se else nn.Identity(),
                nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels, eps=0.001, momentum=0.01),
                )
        else :
            self.layers = nn.Sequential(
                BasicConv(channels, channels, kernel_size=3, groups=channels, activation=activation_layer),
                se_layer(channels, channels // 4) if use_se else nn.Identity(),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels, eps=0.001, momentum=0.01),
                )

    def forward(self, input: Tensor) -> Tensor:
        return input + self.layers(input)


class MobileNetV3(nn.Module):
    def __init__(self,
                 expand_ratio: int = 4,
                 use_se: bool = True,
                 num_re_blocks: int = 2,
                 num_hs_blocks: int = 3,
        ) :
        super().__init__()
        # building inverted residual blocks
        self.body = nn.Sequential(
                *[InvertedResidual(64, expand_ratio, use_se, False) for _ in range(num_re_blocks)],
                *[InvertedResidual(64, expand_ratio, use_se, True) for _ in range(num_hs_blocks)]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)
