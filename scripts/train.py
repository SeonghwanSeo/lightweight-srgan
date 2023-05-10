import sys
sys.path.append(".")
sys.path.append("..")

import torch
import numpy as np
import random
import logging

from src.pretrain import Trainer
from coach.train_manager import Train_Manager
from utils import setup_logger

def main(config) : 
    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    run_dir = config.RUN_DIR
    setup_logger(run_dir)

    if config.restart :
        logging.info(f'Training Restart, Running Directory: {run_dir}')
    else :
        logging.info(f'Training Start, Running Directory: {run_dir}')
    trainer = Trainer(config)
    trainer.fit()

if __name__ == '__main__' :
    coach = Train_Manager()
    config = coach.parse_config()
    main(config)

