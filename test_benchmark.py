import argparse
import os
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data_utils import TestDatasetFromFolder
from model import SRResNet
import utils

TESTDATA_DIR = './data/test_data'
RESULT_DIR = './benchmark_results'

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='MSE_96', type=str, help='generator model epoch name')
parser.add_argument('--step', default=5000, type=int, help='generator model step name')
parser.add_argument('--benchmark', default='Set5', nargs='+', type=str, help='test benchmark', choices=['Set5', 'Set14'])
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name
MODEL_FILE = f'./results_srresnet/{MODEL_NAME}/step_{opt.step}/netG.pth'

model = SRResNet(UPSCALE_FACTOR)
model = model.cuda()
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()

for benchmark in opt.benchmark :
    print(f'Run {MODEL_NAME} - Benchmark {benchmark}')
    DATA_DIR = os.path.join(TESTDATA_DIR, benchmark)
    SAVE_DIR = os.path.join(RESULT_DIR, opt.model_name, benchmark)
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    test_dataset = TestDatasetFromFolder(DATA_DIR, upscale_factor=UPSCALE_FACTOR)
    test_dataloader = DataLoader(test_dataset, num_workers=1)
    with torch.no_grad() :
        for image_name, lr, hr, bicubic_sr in tqdm(test_dataloader):
            lr, hr, bicubic_sr = lr.cuda(), hr.cuda(), bicubic_sr.cuda()
            sr = model(lr)
            image_name = image_name[0]
            save_path = os.path.join(SAVE_DIR, image_name)
            utils.save_image(
                    save_path,
                    lr.squeeze(0),
                    hr.squeeze(0),
                    bicubic_sr.squeeze(0),
                    sr.squeeze(0),
            )
