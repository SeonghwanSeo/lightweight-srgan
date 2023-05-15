import torch
from torch import nn, Tensor

__all__ = ['ShuffleNetV2']

class BasicConv(nn.Sequential) :
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=None,
                 dilation=1,
                 groups=1,
                 activation = True,
        ) :
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if activation else nn.Identity(),
        )

def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class InvertedResidual(nn.Module):
    def __init__(self, channels: int) :
        super().__init__()

        branch_features = channels // 2
        self.branch2 = nn.Sequential(
            BasicConv(branch_features, branch_features, kernel_size=1),
            BasicConv(branch_features, branch_features, kernel_size=3, groups=branch_features, activation=False),
            BasicConv(branch_features, branch_features, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((x1, self.branch2(x2)), dim=1)
        out = channel_shuffle(out, 2)
        return out

class ShuffleNetV2(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        body = nn.Sequential(
            *[InvertedResidual(64) for _ in range(num_blocks)]
        )
        self.body = body

    def forward(self, x) :
        return self.body(x)
