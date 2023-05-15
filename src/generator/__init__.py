import torch
from .base import SRBaseNet
from .body import build_body
from copy import deepcopy

def build_generator(config) -> SRBaseNet :
    body = build_body(config.BODY)
    return SRBaseNet(body)
