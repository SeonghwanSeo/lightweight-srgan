from torchvision.transforms import Compose, ToPILImage, CenterCrop, Resize
import pytorch_ssim
import math
import numpy as np

from matplotlib import pyplot as plt

from typing import Union, List, Tuple, Dict, Optional
from torch import Tensor
from PIL import Image

def calculate_metrics(sr_tensor: Tensor, hr_tensor: Tensor) -> Tuple[float, float, float]:
    sr_tensor, hr_tensor = (sr_tensor+1)/2, (hr_tensor+1)/2
    mse = ((sr_tensor - hr_tensor) ** 2).mean().item()
    psnr = 10 * math.log10(1 / mse)
    ssim = pytorch_ssim.ssim(sr_tensor, hr_tensor).item()
    return mse, ssim, psnr

def tensor2im(var, normalized=False):
    if normalized is False :
        var = ((var + 1) / 2)
    var = var.cpu()
    var[var < 0] = 0
    var[var > 1] = 1
    return ToPILImage()(var)

def plot_image(
        lr_tensor: Tensor,
        hr_tensor: Tensor,
        bicubic_sr_tensor: Tensor,
        model_sr_tensor: Tensor,
        bicubic_metrics: Optional[Dict[str, float]],
        model_metrics: Optional[Dict[str, float]],
        fig, gs, i,
        transform = lambda x: x,
    ) :
    c, h, w = bicubic_sr_tensor.size()
    lr_image = transform(tensor2im(lr_tensor, normalized=True))
    hr_image = transform(tensor2im(hr_tensor))
    bicubic_sr_image = transform(tensor2im(bicubic_sr_tensor))
    model_sr_image = transform(tensor2im(model_sr_tensor))

    fig.add_subplot(gs[i, 0])
    plt.imshow(hr_image)
    plt.title('HR (Ground Truth)')

    fig.add_subplot(gs[i, 1])
    plt.imshow(lr_image)
    plt.title('LR Input')

    fig.add_subplot(gs[i, 2])
    plt.imshow(bicubic_sr_image)
    if bicubic_metrics is None :
        _, ssim, psnr = calculate_metrics(bicubic_sr_tensor, hr_tensor)
    else :
        ssim, psnr = bicubic_metrics['ssim'], bicubic_metrics['psnr']
    plt.title(f'Bicubic Output\npsnr={psnr:.4f}\nssim={ssim:.4f}')

    fig.add_subplot(gs[i, 3])
    plt.imshow(model_sr_image)
    if model_metrics is None :
        _, ssim, psnr = calculate_metrics(model_sr_tensor, hr_tensor)
    else :
        ssim, psnr = model_metrics['ssim'], model_metrics['psnr']
    plt.title(f'Model Output\npsnr={psnr:.4f}\nssim={ssim:.4f}')

def save_image(
        path: str,
        lr_tensor: Tensor,
        hr_tensor: Tensor,
        bicubic_sr_tensor: Tensor,
        model_sr_tensor: Tensor,
        bicubic_metrics: Optional[Dict[str, float]] = None,
        model_metrics: Optional[Dict[str, float]] = None,
    ) :
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 4)
    plot_image(lr_tensor, hr_tensor, bicubic_sr_tensor, model_sr_tensor, bicubic_metrics, model_metrics, fig, gs, 0)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)

def save_images(
        path: str,
        data_list: List[Tuple[Tensor, Tensor, Tensor, Tensor, Optional[dict], Optional[dict]]]
    ) :
    num_images = len(data_list)
    fig = plt.figure(figsize=(16, 4 * num_images))
    gs = fig.add_gridspec(num_images, 4)
    #transform = Compose([Resize(400), CenterCrop(400)])
    for i, data in enumerate(data_list) :
        plot_image(*data, fig, gs, i)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
