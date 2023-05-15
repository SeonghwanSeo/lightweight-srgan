from torch import nn
from .mobilenetv2 import MobileNetV2
from .mobilenetv3 import MobileNetV3
from .resnet import ResNet
from .shufflenetv2 import ShuffleNetV2
from .squeezenet import SqueezeNet
from copy import deepcopy

def build_body(config) -> nn.Module :
    config = deepcopy(config)
    _type = config.pop('_type_')
    if _type == 'ResNet' :
        return ResNet(**config)
    elif _type == 'SqueezeNet' :
        return SqueezeNet(**config)
    elif _type == 'MobileNetV2' :
        return MobileNetV2(**config)
    elif _type == 'MobileNetV3' :
        return MobileNetV3(**config)
    elif _type == 'ShuffleNetV2' :
        return ShuffleNetV2(**config)
    else :
        raise NotImplemented

