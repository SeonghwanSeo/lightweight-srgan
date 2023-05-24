import os
import logging

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision

from typing import Tuple, Dict, List

from . import utils
from .utils import NativeScalerWithGradNormCount
from .generator import build_generator
from .discriminator import VGGStyleDiscriminator
from .dataset import TrainDataset, ValDataset
from .vgg_loss import VGGLoss
from .lpips_loss import LPIPSLoss

try :
    from torch.utils.tensorboard.writer import SummaryWriter
    LOAD_TENSORBOARD = True
except :
    LOAD_TENSORBOARD = False

class Trainer() :
    def __init__(self, config) :
        self.config = config
        self.setup_work_directory()
        self.setup_dataset()
        self.setup_model()
        self.setup_optimizer()
        self.setup_train()

        if config.restart :
            self.load_checkpoint()
        else :
            if config.PRETRAINED_GENERATOR is not None :
                self.load_pretrained_generator()
            if config.PRETRAINED_DISCRIMINATOR is not None :
                self.load_pretrained_discriminator()

    def setup_work_directory(self) :
        run_dir = self.config.RUN_DIR
        self.save_dir = save_dir = os.path.join(run_dir, 'save')
        self.image_log_dir = image_log_dir = os.path.join(run_dir, 'image_log')
        self.tblog_dir = tblog_dir = os.path.join(run_dir, 'tensorboard')
        if not self.config.restart :
            os.mkdir(save_dir)
            os.mkdir(image_log_dir)
            if LOAD_TENSORBOARD :
                os.mkdir(tblog_dir)
        if LOAD_TENSORBOARD :
            self.tb_logger = SummaryWriter(log_dir = self.tblog_dir)

    def setup_dataset(self) :
        logging.info('Setup Data')
        data_cfg = self.config.DATA
        self.train_dataset = TrainDataset(data_cfg.TRAIN_DATA_DIR, data_cfg.CROP_SIZE, 4, data_cfg.RANDOM_RESIZE)
        self.val_dataset = ValDataset(data_cfg.VAL_DATA_DIR)
        self.train_dataloader = DataLoader(
                self.train_dataset,
                data_cfg.TRAIN_BATCH_SIZE,
                shuffle=True,
                num_workers = data_cfg.NUM_WORKERS,
                drop_last = True,
                pin_memory=True
        )
        self.val_dataloader = DataLoader(
                self.val_dataset,
                data_cfg.VAL_BATCH_SIZE,
                shuffle=False,
                num_workers = data_cfg.NUM_WORKERS,
                pin_memory=True
        )
        logging.info(f'num of train data: {len(self.train_dataset)}')
        logging.info(f'num of val data: {len(self.val_dataset)}\n')

    def setup_model(self) :
        logging.info('Setup Model')
        generator_config = self.config.MODEL
        self.generator = build_generator(generator_config)
        self.generator.initialize_weights()
        self.generator.cuda()
        self.discriminator = VGGStyleDiscriminator(self.config.DATA.CROP_SIZE)
        self.discriminator.cuda()
        log = f"number of parameters :\n" + \
              f"Generator     : {sum(p.numel() for p in self.generator.parameters() if p.requires_grad)}\n" + \
              f"Discriminator : {sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)}\n"
        logging.info(log)

    def setup_optimizer(self) :
        train_config = self.config.TRAIN
        self.monitor = train_config.MONITOR
        self.mode = train_config.MODE
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=train_config.LR)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=train_config.LR)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=train_config.LR)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=train_config.LR)
        self.loss_scaler_g = NativeScalerWithGradNormCount(self.generator, self.optimizer_g, train_config.CLIP_GRAD)
        self.loss_scaler_d = NativeScalerWithGradNormCount(self.discriminator, self.optimizer_d, train_config.CLIP_GRAD)
        self.lr_schedular_g = StepLR(self.optimizer_g, train_config.STEP_SIZE, train_config.GAMMA)
        self.lr_schedular_d = StepLR(self.optimizer_d, train_config.STEP_SIZE, train_config.GAMMA)

    def setup_train(self) :
        self.global_step = 0
        self.global_epoch = 0
        if self.mode == 'min' :
            self.best_metric = float('inf')
        else :
            self.best_metric = float('-inf')
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.vgg_loss = VGGLoss('vgg19_54').cuda()
        self.bce_loss_logit = nn.BCEWithLogitsLoss()
        self.lpips_loss = LPIPSLoss('alex').cuda()

    def fit(self) :
        self.generator.train()
        logging.info('Train Start')
        while self.global_step < self.config.MAX_STEP :
            self.global_epoch += 1
            for batch in self.train_dataloader :
                self.run_train_step(batch)
                if self.global_step == self.config.MAX_STEP :
                    break
                if self.global_step % self.config.VAL_INTERVAL == 0 :
                    self.validate()

        self.validate()
        save_path = os.path.join(self.save_dir, 'last.tar')
        self.save_checkpoint(save_path)
        logging.info('Train Finish')

    @torch.no_grad()
    def validate(self) :
        self.generator.eval()
        agg_metric_dict: List[Tuple[int, Dict[str, float]]] = []
        for batch in self.val_dataloader :
            batch_size = batch[0].size(0)
            metric_dict = self.__run_val_step(batch)
            agg_metric_dict.append((batch_size, metric_dict))
        metric_dict = self.aggregate_metric_dict(agg_metric_dict)
        metric = metric_dict[self.monitor]
        update = False
        if self.mode == 'min' :
            if metric < self.best_metric :
                self.best_metric = metric
                update = True
        else :
            if metric > self.best_metric :
                self.best_metric = metric
                update = True
        if update :
            save_path = os.path.join(self.save_dir, 'best.tar')
            self.print_metrics(metric_dict, prefix='VAL* ')
            self.save_checkpoint(save_path)
        else :
            self.print_metrics(metric_dict, prefix='VAL  ')
        self.log_metrics(metric_dict, prefix='VAL')
        self.generator.train()

    def aggregate_metric_dict(self, agg_dict: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        sum_metric = {}
        for batch_size, metric_dict in agg_dict:
            for key, metric in metric_dict.items():
                sum_metric[key] = sum_metric.get(key, 0) + metric * batch_size
        mean_metric_dict = {key: val / len(self.val_dataset) for key, val in sum_metric.items()}
        return mean_metric_dict

    def run_train_step(self, batch: Tuple[Tensor, Tensor]):
        self.global_step += 1
        lr, hr = batch
        lr_ = lr.cuda(non_blocking=True)
        hr_ = hr.cuda(non_blocking=True)

        for p in self.discriminator.parameters() :
            p.requires_grad = False
        self.optimizer_g.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE) :
            sr_ = self.generator(lr_)
            loss_g_perceptual = self.vgg_loss(sr_, hr_)
            loss_g_l1 = self.l1_loss(sr_, hr_)
            logit_sr = self.discriminator(sr_)
            loss_g_gan = self.calc_adversarial_loss(logit_sr, is_real = True)
            loss_g = loss_g_perceptual + 0.01 * loss_g_l1 + 0.001 *loss_g_gan
        self.loss_scaler_g(loss_g)
        self.lr_schedular_g.step()
        sr_ = sr_.detach()

        for p in self.discriminator.parameters() :
            p.requires_grad = True
        self.optimizer_d.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE) :
            logit_sr = self.discriminator(sr_)
            logit_hr = self.discriminator(hr_)
            loss_d_fake = self.calc_adversarial_loss(logit_sr, is_real = False)
            loss_d_real = self.calc_adversarial_loss(logit_hr, is_real = True)
            loss_d = loss_d_fake + loss_d_real
        self.loss_scaler_d(loss_d)
        self.lr_schedular_d.step()

        metrics = {
                'loss_g': loss_g.item(),
                'loss_g_vgg': loss_g_perceptual.item(),
                'loss_g_l1': loss_g_l1.item(),
                'loss_g_gan': loss_g_gan.item(),
                'loss_d': loss_d.item(),
                'loss_d_real': loss_d_real.item(),
                'loss_d_fake': loss_d_fake.item(),
        }
        if self.global_step % self.config.PRINT_INTERVAL == 0 :
            self.print_metrics(metrics, prefix='TRAIN')
        if self.global_step % self.config.TENSORBOARD_INTERVAL == 0 :
            self.log_metrics(metrics, prefix='TRAIN')
        if self.global_step % self.config.IMAGE_INTERVAL == 0 :
            self.log_images(lr, sr_.cpu(), hr)
        if self.global_step % self.config.SAVE_INTERVAL == 0 :
            save_path = os.path.join(self.save_dir, f'ckpt_{self.global_epoch}_{self.global_step}.tar')
            self.save_checkpoint(save_path)

    @torch.no_grad()
    def __run_val_step(self, batch) -> Dict[str, float]:
        lr, hr = batch
        lr = lr.cuda(non_blocking=True)
        hr = hr.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE) :
            sr = self.generator.inference(lr)
            metric_dict = utils.calculate_metrics(sr, hr)
            metric_dict['lpips'] = self.lpips_loss(sr, hr).item()
            
        return metric_dict 

    def calc_adversarial_loss(self, logit, is_real: bool) :
        if is_real :
            target = torch.ones_like(logit)
        else :
            target = torch.zeros_like(logit)
        return self.bce_loss_logit(logit, target)

    def log_metrics(self, metrics, prefix) :
        if LOAD_TENSORBOARD :
            for key, value in metrics.items() :
                self.tb_logger.add_scalars(f'scalar/{key}', {prefix: value}, self.global_step)

    def log_images(self, lr: Tensor, sr: Tensor, hr: Tensor, num_chunks = 4) :
        batch_size = lr.size(0)
        num_image_files = batch_size // num_chunks
        log_dir = os.path.join(self.image_log_dir, str(self.global_step))
        os.mkdir(log_dir)
        for i in range(num_image_files) :
            idx = i * num_chunks
            images = []
            for j in range(idx, idx + num_chunks) :
                _lr = utils.display_transform(lr[j], self.config.DATA.CROP_SIZE)
                _sr, _hr = sr[j], hr[j]
                images.extend([_lr, _sr, _hr])

            image = torchvision.utils.make_grid(images, nrow=3, padding=5)
            save_path = os.path.join(log_dir, f'{i}.png')
            torchvision.utils.save_image(image, save_path, padding=5)

    def print_metrics(self, metrics: Dict[str, float], prefix: str) :
        if prefix == 'TRAIN' :
            log = f'STEP {self.global_step}\t' + \
                f'EPOCH {self.global_epoch}\t' + \
                f'{prefix}\n\t' + \
                f"G_LOSS: {metrics['loss_g']:.4f}\t" + \
                f"G_VGG: {metrics['loss_g_vgg']:.4f}\t" + \
                f"G_L1: {metrics['loss_g_l1']:.4f}\t" + \
                f"G_GAN: {metrics['loss_g_gan']:.4f}\n\t" + \
                f"D_LOSS: {metrics['loss_d']:.4f}\t" + \
                f"D_REAL: {metrics['loss_d_real']:.4f}\t" + \
                f"D_FAKE: {metrics['loss_d_fake']:.4f}\t" + \
                "\n"
        else :
            mse, psnr, ssim, lpips = metrics['mse'], metrics['psnr'], metrics['ssim'], metrics['lpips']
            log = f'STEP {self.global_step}\t' + \
                f'EPOCH {self.global_epoch}\t' + \
                f'{prefix}\n\t' + \
                f'MSE: {mse:.4f}\t' + \
                f'PSNR: {psnr:.4f} dB\t' + \
                f'SSIM: {ssim:.4f}\t' + \
                f'LPIPS: {lpips:.4f}\t' + \
                "\n"
        logging.info(log)

    def save_checkpoint(self, save_path = None):
        save_state = {
                'generator': self.generator.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'optimizer_generator': self.optimizer_g.state_dict(),
                'optimizer_discriminator': self.optimizer_d.state_dict(),
                'lr_schedular_generator': self.lr_schedular_g.state_dict(),
                'lr_schedular_discriminator': self.lr_schedular_d.state_dict(),
                'scaler_generator': self.loss_scaler_g.state_dict(),
                'scaler_discriminator': self.loss_scaler_d.state_dict(),
                'best_metric': self.best_metric,
                'epoch': self.global_epoch,
                'step': self.global_step,
                'config': self.config,
        }
        if save_path is None :
            save_path = os.path.join(self.save_dir, f'ckpt_{self.global_epoch}_{self.global_step}.tar')
        torch.save(save_state, save_path)

    def __find_checkpoint(self) :
        last_step = -1
        ckpt_path = None
        for path in os.listdir(self.save_dir) :
            name, ext = os.path.splitext(path)
            if ext != '.tar' :
                continue
            if name == 'last' :
                ckpt_path = os.path.join(self.save_dir, path)
                break
            if 'ckpt' not in name :
                continue
            step = int(name.split('_')[-1])
            if step > last_step :
                last_step = step
                ckpt_path = os.path.join(self.save_dir, path)
        return ckpt_path

    def load_checkpoint(self):
        ckpt_path = self.__find_checkpoint()
        if ckpt_path is None :
            return

        logging.info(f"==============> Resuming from {ckpt_path}....................")
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        msg = self.generator.load_state_dict(checkpoint['generator'], strict=False)
        msg = self.discriminator.load_state_dict(checkpoint['discriminator'], strict=False)
        logging.info(msg)
        self.optimizer_g.load_state_dict(checkpoint['optimizer_generator'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_discriminator'])
        self.lr_schedular_g.load_state_dict(checkpoint['lr_schedular_generator'])
        self.lr_schedular_d.load_state_dict(checkpoint['lr_schedular_discriminator'])

        self.loss_scaler_g.load_state_dict(checkpoint['scaler_generator'])
        self.loss_scaler_d.load_state_dict(checkpoint['scaler_discriminator'])
        self.global_step = checkpoint['step']
        self.global_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        logging.info(f"=> loaded successfully {ckpt_path} (STEP {checkpoint['step']} EPOCH {checkpoint['epoch']})")

        del checkpoint
        torch.cuda.empty_cache()

    def load_pretrained_generator(self):
        logging.info(f"==============> Loading generator weight {self.config.PRETRAINED_GENERATOR}......")
        checkpoint = torch.load(self.config.PRETRAINED_GENERATOR, map_location='cpu')

        msg = self.generator.load_state_dict(checkpoint['model'], strict=False)
        logging.info(msg)
        logging.info(f"=> loaded successfully {self.config.PRETRAINED_GENERATOR}")

        del checkpoint
        torch.cuda.empty_cache()

    def load_pretrained_discriminator(self):
        logging.info(f"==============> Loading discriminator weight {self.config.PRETRAINED_DISCRIMINATOR}......")
        checkpoint = torch.load(self.config.PRETRAINED_DISCRIMINATOR, map_location='cpu')

        msg = self.discriminator.load_state_dict(checkpoint['model'], strict=False)
        logging.info(msg)
        logging.info(f"=> loaded successfully {self.config.PRETRAINED_DISCRIMINATOR}")

        del checkpoint
        torch.cuda.empty_cache()
