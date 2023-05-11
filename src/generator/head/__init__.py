from torch import nn
from .conv import ConvHead
from .shufflev2 import ShuffleHeadV2
from .mobilev2 import MobileHeadV2
from .mobilev3 import MobileHeadV3
from .squeeze import SqueezeHead
from copy import deepcopy

def build_head(config) -> nn.Module :
    config = deepcopy(config)
    _type = config.pop('_type_')
    if _type == 'ConvHead' :
        return ConvHead(**config)
    elif _type == 'SqueezeHead' :
        return SqueezeHead(**config)
    elif _type == 'MobileHeadV2' :
        return MobileHeadV2(**config)
    elif _type == 'MobileHeadV3' :
        return MobileHeadV3(**config)
    elif _type == 'ShuffleHeadV2' :
        return ShuffleHeadV2(**config)
    else :
        raise NotImplemented


