import torch
from .base import SRBaseNet
from .resnet import SRResNet
from .squeezenet import SRSqueezeNet
from .mobilenetv2 import SRMobileNetV2
from .shufflenetv2 import SRShuffleNetV2
from copy import deepcopy

def build_generator(config) -> SRBaseNet :
    config = deepcopy(config)
    generator_type = config.pop('generator_type')
    if generator_type == 'SRResNet' :
        model = SRResNet(**config)
    elif generator_type == 'SRSqueezeNet' :
        model = SRSqueezeNet(**config)
    elif generator_type == 'SRMobileNetV2' :
        model = SRMobileNetV2(**config)
    elif generator_type == 'SRShuffleNetV2' :
        model = SRShuffleNetV2(**config)
    else :
        raise NotImplemented
    return model
