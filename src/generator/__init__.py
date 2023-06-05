import torch
from .base import SRBaseNet
from .body import build_body
from copy import deepcopy
from omegaconf import OmegaConf, DictConfig
from typing import Tuple

def build_generator(config) -> SRBaseNet :
    body = build_body(config.BODY)
    return SRBaseNet(body)

def load_generator(save_path) -> Tuple[SRBaseNet, DictConfig] :
    ckpt = torch.load(save_path, map_location = 'cpu')
    config = OmegaConf.create(ckpt['config'])
    model_config = config.MODEL
    model_weight = ckpt.get('generator', ckpt.get('model', None))
    assert model_weight is not None
    model = build_generator(model_config)
    assert model is not None
    model.load_state_dict(model_weight)
    del ckpt
    return model, config
