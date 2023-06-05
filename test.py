import os
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor, CenterCrop, Resize, InterpolationMode
from PIL import Image

from src import load_generator
from src.utils import calculate_metrics, display_transform
from src.lpips_loss import LPIPSLoss
import sys

model_path = sys.argv[1]
benchmark_list = ['Set5', 'Set14']
data_dir_list = [f'./data/test_data/{benchmark}/HR/' for benchmark in benchmark_list]
save_dir_list = [f'exp_result/{benchmark}/HR/' for benchmark in benchmark_list]
device: str = 'cpu'


def to_tensor(image: Image.Image):
    tensor = ToTensor()(image)
    if tensor.size(0) == 1:
        tensor = tensor.repeat(3, 1, 1)
    return tensor


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


logging.info(f"==============> Load Weights from {model_path}....................")
model, config = load_generator(model_path)
model.to(device)
lpips_loss = LPIPSLoss('alex')
print(f"=> loaded successfully {model_path}")
model.eval()

inference_time = 0
ssim_list = []
psnr_list = []
lpips_list = []
print('Start')
with torch.no_grad():
    for data_dir, save_dir in zip(data_dir_list, save_dir_list):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print('DATA:', data_dir)
        hr_path_list = [os.path.join(data_dir, x) for x in sorted(os.listdir(data_dir)) if is_image_file(x)]
        for hr_path in hr_path_list:
            # print(hr_path)
            hr_image = Image.open(hr_path)
            image_name = os.path.splitext(os.path.basename(hr_path))[0]
            W, H = hr_image.size
            H_, W_ = (H - H % 4), (W - W % 4)
            h, w = H_ // 4, W_ // 4
            hr_image = CenterCrop((H_, W_))(hr_image)
            lr_image = Resize((h, w), InterpolationMode.BICUBIC)(hr_image)
            bicubic_sr_image = Resize((H_, W_), InterpolationMode.BICUBIC)(lr_image)

            lr, hr, bicubic_sr = to_tensor(lr_image), to_tensor(hr_image), to_tensor(bicubic_sr_image)

            with torch.autocast(device_type=device, enabled=config.AMP_ENABLE):
                st = time.time()
                sr = model(lr.unsqueeze(0)).squeeze(0)
                end = time.time()
                inference_time += (end - st)
            metric = calculate_metrics(sr, hr)
            ssim_list.append(metric['ssim'])
            psnr_list.append(metric['psnr'])
            lpips_list.append(lpips_loss(sr, hr).item())

            lr = display_transform(lr, (H_, W_))
            images = [lr, sr, hr, bicubic_sr]
            image = torchvision.utils.make_grid(images, nrow=4, padding=5)
            save_path = os.path.join(save_dir, f'{image_name}.png')
            torchvision.utils.save_image(image, save_path, padding=5)

        print(f'AVG PSNR : {np.mean(psnr_list):.4f}')
        print(f'AVG SSIM : {np.mean(ssim_list):.4f}')
        print(f'AVG LPIPS: {np.mean(lpips_list):.4f}')
        print(f'AVG TIME: {inference_time / len(hr_path_list):.4f}')
