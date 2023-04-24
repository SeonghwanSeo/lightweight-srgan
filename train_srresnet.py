import argparse
import os
import time
import math
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder
from model import SRResNet
import pytorch_ssim
import utils

RESULT_PATH = './results_srresnet'
#TRAIN_PATH='/scratch/shwan/SRGAN/DIV2K_train_HR'
#VALID_PATH='/scratch/shwan/SRGAN/DIV2K_valid_HR'
TRAIN_PATH='./data/DIV2K_train_HR'
VALID_PATH='./data/DIV2K_valid_HR'

def main(opt) :
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    BATCH_SIZE = opt.batch_size
    MAX_ITERATION = opt.max_iteration

    train_dataset = TrainDatasetFromFolder(TRAIN_PATH, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_dataset = ValDatasetFromFolder(VALID_PATH, upscale_factor=UPSCALE_FACTOR)
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, num_workers=1, batch_size=1, shuffle=False)

    if opt.vgg is not 'None' :
        exp_path = os.path.join(RESULT_PATH, f'{opt.vgg}_{opt.crop_size}')
    else :
        exp_path = os.path.join(RESULT_PATH, f'MSE_{opt.crop_size}')
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    log_dir = os.path.join(exp_path, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tb_logger = SummaryWriter(log_dir = log_dir)
    
    netG = SRResNet(UPSCALE_FACTOR)
    netG.cuda()
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

    epoch = 0
    iteration = 0
    train_st = time.time()
    while iteration < MAX_ITERATION :
        epoch += 1
        running_results = {'g_loss': 0}
        netG.train()
        for lr, hr in train_dataloader:
            """
            lr: [0, 1]
            hr: [-1, 1]
            sr=netG(lr): [-1, 1]
            """
            iteration += 1
            lr, hr = lr.cuda(), hr.cuda()
            netG.zero_grad()
            sr = netG(lr)
            g_loss = mse_loss(sr, hr)
            g_loss.backward()
            optimizerG.step()
            running_results['g_loss'] += g_loss.item()

            if iteration % opt.log_interval == 0 :
                train_end = time.time()
                g_loss = running_results['g_loss'] / iteration
                print(f'[TRAIN]     \t'
                      f'Step {iteration}\t'
                      f'Epoch {epoch}\t'
                      f'Loss_G: {g_loss:.5f}\t'
                      f'Time: {train_end - train_st:.1f}(s)')
                tb_logger.add_scalar(f'scalar/train/g_loss', g_loss, iteration)
                train_st = time.time()

            if iteration % opt.valid_interval == 0 :
                save_path = os.path.join(exp_path, f'step_{iteration}') 
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                torch.save(netG.state_dict(), os.path.join(save_path, 'netG.pth'))
                validate(netG, val_dataloader, save_path, tb_logger, iteration)
                train_st = time.time()

            if iteration == MAX_ITERATION :
                break
    
@torch.no_grad()
def validate(netG, dataloader, save_path, tb_logger, iteration) :
    st = time.time()
    netG.eval()
    valid_results = {'mse': [], 'psnr': [], 'ssim': []}
    val_image_datas = []
    for lr, bicubic_sr, hr in dataloader:
        lr, bicubic_sr, hr = lr.cuda(), bicubic_sr.cuda(), hr.cuda()
        sr = netG(lr)
        mse, ssim, psnr = utils.calculate_metrics(sr, hr)
        valid_results['mse'].append(mse)
        valid_results['ssim'].append(ssim)
        valid_results['psnr'].append(psnr)

        val_image_datas.append(
                [lr.squeeze(0), hr.squeeze(0), bicubic_sr.squeeze(0), sr.squeeze(0), None, {'ssim': ssim, 'psnr': psnr}]
        )
    mse, ssim, psnr = np.mean(valid_results['mse']), np.mean(valid_results['ssim']), np.mean(valid_results['psnr'])
    tb_logger.add_scalar(f'scalar/valid/mse',  mse,  iteration)
    tb_logger.add_scalar(f'scalar/valid/ssim', ssim, iteration)
    tb_logger.add_scalar(f'scalar/valid/psnr', psnr, iteration)

    img_out_path = os.path.join(save_path, 'image')
    if not os.path.exists(img_out_path):
        os.mkdir(img_out_path)
    for index in range(1, math.ceil(len(val_image_datas) // 5) + 1) :
        path = os.path.join(img_out_path, f'index_{index}.png')
        utils.save_images(path, val_image_datas[(index-1)*5 : index*5])
    netG.train()
    end = time.time()
    print(f'[Validation]\t'
          f'PSNR: {psnr:.4f} dB\t'
          f'SSIM: {ssim:.4f}\t'
          f'Time: {end-st:.1f}(s))')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
    parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                        help='super resolution upscale factor')
    parser.add_argument('--batch_size', default=16, type=int, help='train batch size')
    parser.add_argument('--valid_interval', default=500, type=int, help='valid test interval')
    parser.add_argument('--log_interval', default=100, type=int, help='log interval')
    parser.add_argument('--max_iteration', default=1e6, type=int, help='valid test interval')
    parser.add_argument('--vgg', default='None', type=str, help='vgg type',
                        choices=['None', 'vgg19_22', 'vgg19_54', 'vgg16_22', 'vgg16_53'])
    opt = parser.parse_args()
    main(opt)
