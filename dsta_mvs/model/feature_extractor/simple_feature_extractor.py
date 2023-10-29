from typing import Sequence, Tuple

from torch import nn, Tensor

from ..common import RELU_TYPE, NORM2D_TYPE
from ..common import common_modules as cm

from ..pos_encoding import PositionalEncoder


class SimpleFeatExtraction(nn.Module):
    def __init__(
        self,
        in_size: Tuple[int, int],
        in_chs = 3,
        chs: int = 8,
        k_sz: int = 3,
        layers: Sequence[int] = [5, 10],
        norm_type: str = 'batch',
        relu_type: str = 'leaky',
    ):
        super().__init__()

        self.chs         = chs
        self.k_sz        = k_sz
        self.layers      = layers
        self.num_steps   = len(self.layers)
        self.in_size = in_size

        # Normalization layer class type.
        self.norm_type = NORM2D_TYPE[norm_type]
        self.relu_type = RELU_TYPE[relu_type]
        
        self.first = cm.BaseConvBlk2d(
            in_chs = in_chs,
            out_chs = self.chs,
            kernel_size = 5,
            stride=2, 
            activation = self.relu_type(),
            norm_layer = self.norm_type(self.chs)
        )

        self.blks = self._build()
        
        # calculate input size for SphereConvBlk
        size = self.first.infer_size(self.in_size)
        for layer in self.blks:
            size = layer.infer_size(size)

        # Originally does not have activation and normalization layer.
        self.final_layer = cm.BaseConvBlk2d(
            in_chs = self.chs,
            out_chs = self.chs,
            kernel_size = self.k_sz, 
            activation = self.relu_type(),
            norm_layer = self.norm_type(self.chs)
        )

    def _build(self):

        blks = []

        for i in range(self.num_steps):
            for _ in range(self.layers[i]):
                blks.append(
                    cm.ResConvBlk2d(
                        in_chs = self.chs,
                        out_chs = self.chs,
                        kernel_size = self.k_sz,
                        activation = self.relu_type(),
                        norm_layer = self.norm_type(self.chs)))

            if i != self.num_steps-1:
                blks.append(cm.BaseConvBlk2d(
                    in_chs = self.chs,
                    out_chs = self.chs,
                    kernel_size = 3,
                    stride = 2,
                    activation = self.relu_type(), 
                    norm_layer = self.norm_type(self.chs) ) )

        return nn.Sequential(*blks)

    def forward(self, x:Tensor):
        x = self.first(x)
        x = self.blks(x)
        return self.final_layer(x)
    

class SimpleFeatExtractionWPosEncoding(
    SimpleFeatExtraction
):
    def __init__(
        self,
        in_size: Tuple[int, int],
        chs: int = 8,
        k_sz: int = 3,
        layers: Sequence[int] = [5, 10],
        norm_type: str = 'batch',
        relu_type: str = 'leaky',
        c_dim: int = 2,
        coord_dims: Tuple[int, int] = None,
        coord_encoder: nn.Module = None
    ):

        super().__init__(
            in_size=in_size,
            in_chs=3 + coord_encoder.out_dim*len(coord_dims),
            chs=chs,
            k_sz=k_sz,
            layers=layers,
            norm_type=norm_type,
            relu_type=relu_type
        )

        self.pos_encoder = PositionalEncoder(
            c_dim, coord_dims, coord_encoder
        )


    def forward(self, x:Tensor):
        x = self.pos_encoder(x)
        x = self.first(x)
        x = self.blks(x)
        return self.final_layer(x)