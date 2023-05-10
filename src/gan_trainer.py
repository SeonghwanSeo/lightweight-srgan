import os
import logging
from omegaconf import OmegaConf

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from typing import Tuple, Dict, List

from .utils import NativeScalerWithGradNormCount, calculate_metrics
from .generator import build_generator
from .discriminator import VGGStyleDiscriminator
from .dataset import TrainDataset, ValDataset

try :
    from torch.utils.tensorboard.writer import SummaryWriter
    LOAD_TENSORBOARD = True
except :
    LOAD_TENSORBOARD = False

class Trainer() :
    def __init__(self, args, run_dir: str) :
        self.args = args
        self.setup_trainer()
        self.setup_work_directory(run_dir)
        self.setup_dataset()
        self.setup_model()
        self.setup_optimizer()

    def setup_trainer(self) :
        logging.info('Setup Trainer')
        args = self.args
        self.fp16 = args.fp16
        self.model_config = OmegaConf.load(args.model_config)
        self.train_config = OmegaConf.load(args.train_config)
        self.max_step = args.max_step

        self.num_workers = args.num_workers
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size

        self.val_interval = args.val_interval
        self.save_interval = args.save_interval
        self.log_interval = args.log_interval
        self.print_interval = args.print_interval

    def setup_work_directory(self, run_dir) :
        logging.info('Setup Work Directory')
        self.save_dir = save_dir = os.path.join(run_dir, 'save')
        self.log_dir = log_dir = os.path.join(run_dir, 'log')
        self.model_config_path = os.path.join(run_dir, 'model_config.yaml')
        self.train_config_path = os.path.join(run_dir, 'train_config.yaml')
        os.mkdir(save_dir)
        os.mkdir(log_dir)
        if LOAD_TENSORBOARD :
            tblog_dir = os.path.join(run_dir, 'tensorboard')
            os.mkdir(tblog_dir)
            self.tb_logger = SummaryWriter(log_dir = tblog_dir)

    def setup_dataset(self) :
        logging.info('Setup Data')
        self.train_dataset = TrainDataset(args.train_data_dir)
        self.val_dataset = ValDataset(args.val_data_dir)
        self.train_dataloader = DataLoader(self.train_dataset, args.train_batch_size, \
                                    num_workers = args.num_workers, shuffle = True,
                                    drop_last = True, pin_memory=True)
           
        self.val_dataloader = DataLoader(self.val_dataset, args.val_batch_size, \
                                    num_workers = args.num_workers, pin_memory=True)
        logging.info(f'num of train data: {len(self.train_dataset)}')
        logging.info(f'num of val data: {len(self.val_dataset)}\n')

    def setup_model(self) :
        logging.info('Setup Model')
        model_config = self.model_config
        generator = build_generator(model_config)
        discriminator = VGGStyleDiscriminator()
        self.generator = generator.cuda()
        self.discriminator = discriminator.cuda()
        self.generator.initialize_weights(self.args.pretrained_model)
        self.discriminator.initialize_weights()
        log = f"number of parameters :\n" 
        log += f"Generator      : {sum(p.numel() for p in self.generator.parameters() if p.requires_grad)}\n"
        log += f"Discriminator  : {sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)}\n"
        logging.info(log)

    def setup_optimizer(self) :
        train_config = self.train_config
        OmegaConf.save(train_config, self.train_config_path)
        self.monitor = train_config.monitor
        self.mode = train_config.mode
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=train_config.lr)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=train_config.lr)
        self.loss_scaler = NativeScalerWithGradNormCount(self.generator, self.optimizer_g, train_config.clip_grad)
        self.loss_scaler = NativeScalerWithGradNormCount(self.discriminator, self.optimizer_d, train_config.clip_grad)
        self.mse_loss = torch.nn.MSELoss()

    def fit(self) :
        self.global_step = 0
        self.global_epoch = 0
        if self.mode == 'min' :
            self.best_metric = float('inf')
        else :
            self.best_metric = float('-inf')
        self.model.train()
        self.optimizer.zero_grad()
        logging.info('Train Start')
        while self.global_step < self.max_step :
            self.global_epoch += 1
            for batch in self.train_dataloader :
                metrics = self.run_train_step(batch)

                self.global_step += 1
                if self.global_step % self.log_interval == 0 :
                    self.log_metrics(metrics, prefix='TRAIN')
                if self.global_step % self.print_interval == 0 :
                    self.print_metrics(metrics, prefix='TRAIN')
                if self.global_step % self.save_interval == 0 :
                    save_path = os.path.join(self.save_dir, f'ckpt_{self.global_epoch}_{self.global_step}.tar')
                    self.model.save(save_path)
                if self.global_step == self.max_step :
                    break
                if self.global_step % self.val_interval == 0 :
                    self.validate()

        self.validate()
        save_path = os.path.join(self.save_dir, 'last.tar')
        self.model.save(save_path)
        logging.info('Train Finish')

    @torch.no_grad()
    def validate(self) :
        self.model.eval()
        agg_metric_dict: List[Tuple[int, Dict[str, float]]] = []
        for batch in self.val_dataloader :
            batch_size = batch[0].size(0)
            metric_dict = self.run_val_step(batch)
            agg_metric_dict.append((batch_size, metric_dict))
        metric_dict = self.aggregate_metric_dict(agg_metric_dict)
        metric = metric_dict[self.monitor]
        update = False
        if self.mode == 'min' :
            if metric < self.best_metric :
                self.best_metric = best_metric
                update = True
        else :
            if metric > self.best_metric :
                self.best_metric = best_metric
                update = True
        if update :
            save_path = os.path.join(self.save_dir, 'best.tar')
            self.model.save(save_path)
            self.print_metrics(metric_dict, prefix='VAL* ')
        else :
            self.print_metrics(metric_dict, prefix='VAL  ')
        self.log_metrics(metric_dict, prefix='VAL')
        self.model.train()

    def aggregate_metric_dict(self, agg_dict: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        sum_metric = {}
        for batch_size, metric_dict in agg_dict:
            for key, metric in metric_dict.items():
                sum_metric[key] = sum_metric.get(key, 0) + metric * batch_size
        mean_metric_dict = {key: val / len(self.val_dataset) for key, val in sum_metric.items()}
        return mean_metric_dict

    def run_train_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, float]:
        lr, hr = batch
        lr = lr.cuda(non_blocking=True)
        hr = hr.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.fp16) :
            sr = self.model(lr)
            loss = self.mse_loss(sr, hr)
        self.loss_scaler(loss)
        self.optimizer.zero_grad()
        self.lr_schedular.step_update(self.global_step)
        return {'mse': loss.item()}

    @torch.no_grad()
    def run_val_step(self, batch) -> Dict[str, float]:
        lr, hr = batch
        lr = lr.cuda(non_blocking=True)
        hr = hr.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.fp16) :
            sr = self.model(lr)
        metric_dict = calculate_metrics(sr, hr)
        return metric_dict 

    def log_metrics(self, metrics, prefix) :
        if LOAD_TENSORBOARD :
            for key, value in metrics.items() :
                self.tb_logger.add_scalars(f'scalar/{key}', {prefix: value}, self.global_step)

    def print_metrics(self, metrics: Dict[str, float], prefix: str) :
        if prefix == 'TRAIN' :
            mse = metrics['mse']
            log = f'STEP {self.global_step}\t' + \
                f'EPOCH {self.global_epoch}\t' + \
                f'{prefix}\t' + \
                f'mse: {mse:.4f}\t' + \
        else :
            mse, psnr, ssim = metrics['mse'], metrics['psmr'], metrics['ssim']
            log = f'STEP {self.global_step}\t' + \
                f'EPOCH {self.global_epoch}\t' + \
                f'{prefix}\t' + \
                f'mse: {mse:.4f}\t' + \
                f'psnr: {psnr:.4f} dB\t'
                f'ssim: {ssim:.4f}'
        logging.info(log)
