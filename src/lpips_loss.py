import lpips
from torch import nn

class LPIPSLoss(nn.Module):
    def __init__(self, net: str = 'alex'):
        super(LPIPSLoss, self).__init__()
        self.loss_fn = lpips.LPIPS(net=net)

    def forward(self, sr, hr):
        sr = sr * 2 - 1
        hr = hr * 2 - 1

        return self.loss_fn(sr, hr)
