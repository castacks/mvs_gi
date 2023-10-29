import copy
from typing import Sequence

from torch import nn, Tensor
# import torch.nn.functional as F

from ..common import RELU_TYPE, NORM3D_TYPE
from ..common import common_modules as cm

# General structure of a UNet in this file.
# x -> D0 ------------------------> + -> out_costs -> output
#      |                            |
#      |-> D1 ----------> + -> U1 ->|
#          |              |
#          |-> D2 -> U0 ->|
class UNetCostVolumeRegulatorBase(nn.Module):
    def __init__(
        self, 
        in_chs: int=16,
        f_int_chs: int=32, # The first intermediate channels in the UNet.
        final_chs: int=1,
        u_depth: int=3,
        blk_width: int=4,
        stage_factor: int=2,
        cost_k_sz: int=3,
        keep_last_chs: Sequence[int]=[],
        deconv_k_sz: int=3,
        norm_type: str = 'batch',
        relu_type: str = 'leaky'
    ):
        super().__init__()
        self.in_chs          = in_chs
        self.f_int_chs       = f_int_chs
        self.final_chs       = final_chs
        self.u_depth         = u_depth
        self.blkWidth        = blk_width
        self.stage_factor    = stage_factor
        self.k_sz            = cost_k_sz
        self.keep_last_chs   = keep_last_chs
        self.deconv_k_sz     = deconv_k_sz

        self.norm_type = NORM3D_TYPE[norm_type]
        self.relu_type = RELU_TYPE[relu_type]

        self._int_chs = [] # The intermediate channels of the down block outputs.

        self.down_blks = self.build_down()
        self.upBlks = self.build_up()

        # self._int_chs has already been populated.
        assert len(self._int_chs) > 0, "The down blocks must be built first."
        self.out_costs = nn.Sequential(
            cm.ResizeConv3d(
                self._int_chs[0],
                self.in_chs, # TODO: Will need to be changed to something like self._int_chs[0].
                self.deconv_k_sz,
                stride=2, 
                activation=self.relu_type(),
                norm_layer=self.norm_type(self.in_chs),
            ), # TODO: Will need to add an additional layer here to properly reduce the channels.
            cm.BaseConvBlk3d(
                self.in_chs, # TODO: Will need to be changed to something like self._int_chs[0].
                self.final_chs,
                self.deconv_k_sz,
                bias_on=True,
                activation=cm.NoOp(),  
                norm_layer=cm.NoOp(),
            )
        )

    def build_down(self):
        self._int_chs = []
        down_blks = []
        temp_in_chs, temp_out_chs = self.in_chs, self.f_int_chs
        for i in range(self.u_depth):
            if i not in self.keep_last_chs:
                if i != 0:
                    temp_out_chs *= self.stage_factor

            down_blks.append(
                UNetDownBlk(
                    temp_in_chs,
                    temp_out_chs,
                    self.k_sz, 
                    self.blkWidth,
                    activation = self.relu_type(),
                    norm_layer = self.norm_type(temp_out_chs) ) )
            
            self._int_chs.append(temp_out_chs)
            
            temp_in_chs = temp_out_chs

        return nn.ModuleList(down_blks)

    def build_up(self):
        assert len(self._int_chs) > 0, "The down blocks must be built first."
        # Reverse self._int_chs.
        int_chs = copy.deepcopy(self._int_chs)
        int_chs.reverse()
        
        up_deconvs = []

        for i in range( self.u_depth - 1 ):
            temp_in_chs  = int_chs[ i ]
            temp_out_chs = int_chs[ i + 1 ]

            up_deconvs.append(
                cm.ResizeConv3d(
                    temp_in_chs,
                    temp_out_chs,
                    self.deconv_k_sz,
                    stride = 2,
                    activation = self.relu_type(),
                    norm_layer = self.norm_type(temp_out_chs),
                )
            )

        return nn.ModuleList(up_deconvs)

    def forward(self, x: Tensor):
        '''x: cost volume tensor of size [B, C, N, H, W]
        '''

        # Keep passing the input through the first block and then down the Cost Volume.
        # Save the residual for the ascending side in res_list.
        res_list = []
        for i, down_blk in enumerate(self.down_blks):
            x = down_blk(x)

            if i != self.u_depth-1:
                res_list.append(x)

        # Reverse the list of residuals to align the tensors for the ascending side.
        res_list.reverse()

        #P ass the input through the up blocks and add the res_list residual.
        for i, up_blk in enumerate(self.upBlks):
            x = up_blk(x, res=res_list[i])

        return self.out_costs(x)

class UNetCostVolumeRegulator(nn.Module):
    def __init__(
        self, 
        in_chs: int,
        final_chs: int,
        u_depth: int,
        blk_width: int,
        stage_factor: int,
        cost_k_sz: int,
        keep_last_chs: Sequence[int],
        deconv_k_sz: int,
        sweep_fuse_ch_reduce: int = 2,
        num_cams: int = 3,
	    only_one_cam: bool = False,
        norm_type: str = 'batch',
        relu_type: str = 'leaky'
    ):
        super().__init__()
        # TODO: The use of only_one_cam is not in the intended way in the current configuration.
        self.in_chs          = ( in_chs * num_cams ) // sweep_fuse_ch_reduce
        self.in_chs          = in_chs if only_one_cam else self.in_chs
        self.final_chs       = final_chs
        self.u_depth         = u_depth
        self.blkWidth        = blk_width
        self.stage_factor    = stage_factor
        self.k_sz            = cost_k_sz
        self.keep_last_chs   = keep_last_chs
        self.deconv_k_sz     = deconv_k_sz

        self.norm_type = NORM3D_TYPE[norm_type]
        self.relu_type = RELU_TYPE[relu_type]

        self.down_blks = self.build_down()
        self.upBlks = self.build_up()

        self.out_costs = nn.Sequential(
            cm.ResizeConv3d(
                2*self.in_chs,
                self.in_chs,
                self.deconv_k_sz,
                stride=2, 
                activation=self.relu_type(),
                norm_layer=self.norm_type(self.in_chs),
            ),
            cm.BaseConvBlk3d(
                self.in_chs,
                self.final_chs,
                self.deconv_k_sz,
                bias_on=True,
                activation=cm.NoOp(),  
                norm_layer=cm.NoOp(),
            )
        )

    def build_up(self):
        
        up_deconvs = []
        t_chs = self.in_chs * (
            self.stage_factor**( self.u_depth - len(self.keep_last_chs) ) )
        
        temp_in_chs, temp_out_chs = t_chs, t_chs 

        for i in range(self.u_depth-1):
            if i+1 not in self.keep_last_chs:
                temp_out_chs = temp_out_chs // self.stage_factor

            up_deconvs.append(
                cm.ResizeConv3d(
                    temp_in_chs,
                    temp_out_chs,
                    self.deconv_k_sz,
                    stride = 2,
                    activation = self.relu_type(),
                    norm_layer = self.norm_type(temp_out_chs),
                )
            )

            if i+1 not in self.keep_last_chs:
                temp_in_chs = temp_in_chs // self.stage_factor

        return nn.ModuleList(up_deconvs)

    def build_down(self):

        down_blks = []
        temp_in_chs, temp_out_chs = self.in_chs, self.in_chs
        for i in range(self.u_depth):
            if i not in self.keep_last_chs:
                temp_out_chs *= self.stage_factor

            down_blks.append(
                UNetDownBlk(
                    temp_in_chs,
                    temp_out_chs,
                    self.k_sz, 
                    self.blkWidth,
                    activation = self.relu_type(),
                    norm_layer = self.norm_type(temp_out_chs) ) )
            
            if i not in self.keep_last_chs:
                temp_in_chs *= self.stage_factor

        return nn.ModuleList(down_blks)

    #Pre: Feat 3D Volume that has been aligned and features have been extracted
    #Format: (B, N, C, H, W)
    def forward(self, x: Tensor):

        #Keep passing the input through the first block and then down the Cost Volume.
        #Save the residual for the ascending side in res_list.
        res_list = []
        for i, down_blk in enumerate(self.down_blks):
            x = down_blk(x)

            if i != self.u_depth-1:
                res_list.append(x)

        #Reverse the list of residuals to align the tensors for the ascending side
        res_list.reverse()

        #Pass the input through the ConvTranspose3d blocks and add the res_list residual
        for i, up_blk in enumerate(self.upBlks):
            x = up_blk(x, res=res_list[i])
            
        #Pass the output through the outCosts ConvTranspose3d to get a tensor of size 
        #[Bat_chSize, 1, N/2, Hout, Wout] where dim=1 is a pre-softmax confidence value 
        #for each depth candidate

        return self.out_costs(x)

#Descending side of U-Net for the Cost Volume 
class UNetDownBlk(nn.Module):
    def __init__(self, 
        in_chs: int,
        out_chs: int,
        kernel_size: int, 
        width: int,
        activation: nn.Module = cm.NoOp(),
        norm_layer: nn.Module = cm.NoOp()
    ):
        super().__init__()

        self.k_sz = kernel_size
        
        self.first = cm.BaseConvBlk3d(
            in_chs,
            out_chs,
            self.k_sz,
            stride = 2,
            activation = copy.deepcopy(activation),
            norm_layer = copy.deepcopy(norm_layer),
        )

        self.blks = [ cm.ResConvBlk3d(
                out_chs,
                out_chs,
                self.k_sz,
                activation = copy.deepcopy(activation),
                norm_layer = copy.deepcopy(norm_layer),
            ) for _ in range(width-1) ]

        self.blks = nn.Sequential(*self.blks)

    def forward(self, x):

        x = self.first(x)
        x = self.blks(x)

        return x