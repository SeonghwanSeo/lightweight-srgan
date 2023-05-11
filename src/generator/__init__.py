import torch
from .base import SRBaseNet
from .body import build_body
from .head import build_head
from copy import deepcopy

def build_generator(config) -> SRBaseNet :
    body = build_body(config.BODY)
    head = build_head(config.HEAD)

    return SRBaseNet(body, head)
