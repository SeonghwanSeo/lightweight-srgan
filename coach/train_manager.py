import argparse
import os
from omegaconf import OmegaConf
from pathlib import Path

__all__ = ['Train_Manager']

class Train_ArgParser(argparse.ArgumentParser) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.formatter_class=argparse.ArgumentDefaultsHelpFormatter

        # Experiment information
        exp_args = self.add_argument_group('experiment information')
        exp_args.add_argument('--name', type=str, help='job name', required=True)
        exp_args.add_argument('--exp_dir', type=str, help='path of experiment directory', default='./result/')
        exp_args.add_argument('--resume', action='store_true', help='Resume Training.')

        # config
        cfg_args = self.add_argument_group('config')
        cfg_args.add_argument('--config', type=str, help='path to config file', required=True)
        cfg_args.add_argument('--model_config', type=str, help='path to model config file') 
        cfg_args.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs='+',
        )

        # easy config modification
        modif = self.add_argument_group('config modification')
        modif.add_argument('--seed', type=int, help='Seed')
        modif.add_argument('--train_data_dir', type=str, help='dataset directory')
        modif.add_argument('--val_data_dir', type=str, help='dataset directory')
        modif.add_argument('--train_batch_size', type=int, help='Training Batch Size')
        modif.add_argument('--val_batch_size', type=int, help='Validation Batch Size')
        modif.add_argument('--num_workers', type=int, help='Number of Dataloader Workers')

        modif.add_argument('--crop_size', type=int, help='Size of Image Patch (HR)')
        modif.add_argument('--random_resize', action='store_true', help='Data Augmentation')

        modif.add_argument('--amp', action='store_true', help='Train With Mixed Prediction')

        modif.add_argument('--pretrained_generator', type=str, help='path of pretrained generator')
        modif.add_argument('--pretrained_discriminator', type=str, help='path of pretrained discriminator')

        modif.add_argument('--max_step', type=int, help='Max Step')
        modif.add_argument('--val_interval', type=int, help='Valiation Interval(Step)')
        modif.add_argument('--print_interval', type=int, help='Printing Interval(Step)')
        modif.add_argument('--tensorboard_interval', type=int, help='Tensorboard Logging Interval(Step)')
        modif.add_argument('--image_interval', type=int, help='Logging Interval(Stp)')
        modif.add_argument('--save_interval', type=int, help='Model Checkpoint Interval(Step)')

class Train_Manager() :
    def __init__(self) :
        self.parser = Train_ArgParser()

    def parse_config(self) :
        args = self.parser.parse_args()
        run_dir = os.path.join(args.exp_dir, args.name)
        if args.resume :
            restart = os.path.exists(run_dir)
            Path(run_dir).mkdir(parents=True, exist_ok=True)
        else :
            assert not os.path.exists(run_dir), f'{run_dir} is already exsits. If you want to resume training, use `--resume`'
            restart = False
            Path(run_dir).mkdir(parents=True, exist_ok=False)

        config_path = os.path.join(run_dir, 'config.yaml')
        if restart :
            restart_config = OmegaConf.load(config_path)
            base_config = OmegaConf.load(args.config)
            config = OmegaConf.merge(base_config, restart_config)
            config.restart = True
        else :
            config = OmegaConf.load(args.config)
            config.RUN_DIR = run_dir
            model_config = OmegaConf.load(args.model_config)
            config.MODEL = model_config

        if args.opts is not None :
            assert len(args.opts) % 2 == 0
            for full_key, v in zip(args.opts[0::2], args.opts[1::2]):
                _cfg = config
                key_list = full_key.split('.')
                for subkey in key_list[:-1]:
                    assert subkey in _cfg
                    _cfg = _cfg[subkey.upper()]
                subkey = key_list[-1].upper()
                assert subkey in _cfg
                _cfg[subkey] = v

        if args.seed is not None :
            config.SEED = args.seed
        if args.amp :
            config.AMP_ENABLE = True
        if args.pretrained_generator is not None :
            config.PRETRAINED_GENERATOR = args.pretrained_generator
        if args.pretrained_discriminator is not None :
            config.PRETRAINED_DISCRIMINATOR = args.pretrained_discriminator
        
        if args.train_data_dir is not None :
            config.DATA.TRAIN_DATA_DIR= args.train_data_dir
        if args.val_data_dir is not None :
            config.DATA.VAL_DATA_DIR= args.val_data_dir
        if args.train_batch_size is not None :
            config.DATA.TRAIN_BATCH_SIZE = args.train_batch_size
        if args.val_batch_size is not None :
            config.DATA.VAL_BATCH_SIZE = args.val_batch_size
        if args.num_workers is not None :
            config.DATA.NUM_WORKERS = args.num_workers

        if args.random_resize :
            config.DATA.RANDOM_RESIZE = True
        if args.crop_size is not None :
            config.DATA.CROP_SIZE = args.crop_size

        if args.max_step is not None :
            config.MAX_STEP = args.max_step
        if args.val_interval is not None :
            config.VAL_INTERVAL = args.val_interval
        if args.print_interval is not None :
            config.PRINT_INTERVAL = args.print_interval
        if args.tensorboard_interval is not None :
            config.TENSORBOARD_INTERVAL = args.tensorboard_interval
        if args.image_interval is not None :
            config.IMAGE_INTERVAL = args.image_interval
        if args.save_interval is not None :
            config.SAVE_INTERVAL = args.save_interval

        if not restart :
            OmegaConf.save(config, config_path)
            config.restart = False

        return config
