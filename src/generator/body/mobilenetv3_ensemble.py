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
            self.hidden_dim = hidden_dim = channels * expand_ratio
            self.expand_conv = BasicConv(channels, hidden_dim, kernel_size=1, activation=activation_layer)
        else :
            self.hidden_dim = hidden_dim = channels
            self.expand_conv = nn.Identity()

        hidden_size1x1 = int(hidden_dim * 1/2)
        hidden_size3x3 = int(hidden_dim * 1/4)
        hidden_size5x5 = int(hidden_dim * 1/8)
        hidden_size7x7 = int(hidden_dim * 1/8)
        self.channel_splits = (hidden_size1x1, hidden_size1x1 + hidden_size3x3, hidden_size1x1 + hidden_size3x3 + hidden_size5x5)
        self.conv3x3 = nn.Conv2d(hidden_size3x3, hidden_size3x3, kernel_size=3, padding = 1, groups=hidden_size3x3)
        self.conv5x5 = nn.Conv2d(hidden_size5x5, hidden_size5x5, kernel_size=5, padding = 2, groups=hidden_size5x5)
        self.conv7x7 = nn.Conv2d(hidden_size7x7, hidden_size7x7, kernel_size=7, padding = 3, groups=hidden_size7x7)

        self.batch_norm = nn.BatchNorm2d(hidden_dim, eps=0.001, momentum=0.01)
        self.activation = activation_layer(inplace=True)

        if use_se :
            self.se_layer = se_layer(hidden_dim, hidden_dim // 4)
        else :
            self.se_layer = nn.Identity()

        self.pointwise = nn.Sequential(
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels, eps=0.001, momentum=0.01),
        )

    def forward(self, input: Tensor) -> Tensor:
        x = self.expand_conv(input)
        x1, x2, x3, x4 = torch.tensor_split(x, self.channel_splits, dim=1)
        xs = [x1, self.conv3x3(x2), self.conv5x5(x3), self.conv7x7(x4)]
        x = torch.cat(xs, dim=1)
        x = self.activation(self.batch_norm(x))
        x = self.pointwise(self.se_layer(x))
        return input + x


class EnsembleMobileNetV3(nn.Module):
    def __init__(self,
                 expand_ratio: int = 4,
                 use_se: bool = True,
                 num_re_blocks: int = 8,
                 num_hs_blocks: int = 8,
        ) :
        super().__init__()
        # building inverted residual blocks
        self.body = nn.Sequential(
                *[InvertedResidual(64, expand_ratio, use_se, False) for _ in range(num_re_blocks)],
                *[InvertedResidual(64, expand_ratio, use_se, True) for _ in range(num_hs_blocks)]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)

