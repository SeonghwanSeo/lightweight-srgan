import torch
from torch._six import inf
from torchvision.transforms import ToPILImage, Compose, ToTensor, Resize, InterpolationMode
import math
from typing import Dict
from torch import Tensor

from . import pytorch_ssim

def display_transform(tensor: Tensor, size: int) -> Tensor:
    tensor = tensor.cpu()
    torch.clamp_(tensor, 0, 1)
    if tensor.size()[-1] == size :
        return tensor
    else :
        transform = Compose([
            ToPILImage(), Resize(size, InterpolationMode.NEAREST), ToTensor()
        ])
        return transform(tensor)

def calculate_metrics(sr_tensor: Tensor, hr_tensor: Tensor) -> Dict[str, float]:
    mse = ((sr_tensor - hr_tensor) ** 2).mean().item()
    psnr = 10 * math.log10(1. / mse)
    ssim = pytorch_ssim.ssim(sr_tensor, hr_tensor).item()
    return dict(mse = mse, ssim = ssim, psnr = psnr)


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

# https://github.com/microsoft/Swin-Transformer
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"
    def __init__(self, model, optimizer, clip_grad):
        self.model = model
        self.optimizer = optimizer
        self.clip_grad = clip_grad
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss):
        create_graph = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if self.clip_grad is not None:
            self._scaler.unscale_(self.optimizer)  # unscale the gradients of optimizer's assigned params in-place
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        else:
            self._scaler.unscale_(self.optimizer)
            ampscaler_get_grad_norm(self.model.parameters())
        self._scaler.step(self.optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

