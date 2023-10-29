from typing import Dict, Type

from torch import nn

from .common_modules import NoOp

RELU_TYPE: Dict[str, Type[nn.Module]] = {
    'original': nn.ReLU,
    'leaky': nn.LeakyReLU,
    'none': NoOp,
}

NORM2D_TYPE: Dict[str, Type[nn.Module]] = {
    'batch': nn.BatchNorm2d,
    'instance': nn.InstanceNorm2d,
    'none': NoOp,
}

NORM3D_TYPE: Dict[str, Type[nn.Module]] = {
    'batch': nn.BatchNorm3d,
    'instance': nn.InstanceNorm3d,
    'none': NoOp,
}