import torch
from torch import nn
from torchvision.models.vgg import vgg16, vgg19
from torchvision.models.vgg import VGG16_Weights, VGG19_Weights

class VGGLoss(nn.Module):
    VGG_VER = {
            'vgg19_22': (vgg19, VGG19_Weights.DEFAULT, 9),
            'vgg19_54': (vgg19, VGG19_Weights.DEFAULT, 36),
            'vgg16_22': (vgg16, VGG16_Weights.DEFAULT, 9),
            'vgg16_53': (vgg16, VGG16_Weights.DEFAULT, 30),
    }
    def __init__(self, vgg: str = 'vgg19_54'):
        super(VGGLoss, self).__init__()
        vgg_arch, weights, num_layers = self.VGG_VER[vgg]
        vgg_model = vgg_arch(weights=weights)
        self.vgg = nn.Sequential(*list(vgg_model.features)[:num_layers]).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr, hr):
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std

        return self.mse_loss(self.vgg(sr), self.vgg(hr))
