import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..common import RELU_TYPE, NORM3D_TYPE
from ..common import common_modules as cm
from ..backports import bilinear_grid_sample as grid_sample

class SphericalSweep(nn.Module):

    def __init__(
        self,
        num_cams: int,
        feat_chs: int,
        post_k_sz: int,
        norm_type: str = 'batch',
        relu_type: str = 'leaky'
    ):
        super().__init__()

        self.num_cams   = num_cams
        self.feat_chs   = feat_chs

        self.norm_type = NORM3D_TYPE[norm_type]
        self.relu_type = RELU_TYPE[relu_type]

        # For grid sample.
        self.grid_sample_mode = 'bilinear'    

        self.post_vol = cm.BaseConvBlk3d(
            in_chs = self.feat_chs,
            out_chs = self.feat_chs,
            kernel_size = post_k_sz,
            activation=self.relu_type(), 
            norm_layer=self.norm_type(self.feat_chs)
        )

    def sweep(self, feats:Tensor, grids:Tensor, masks:Tensor):
        '''
        feats: [B, C#, C, Hi, Wi] .
        grids: [B, C#, X, Ho, Wo, 2], where X is the number of inv dist candidates.
        masks: [B, C#, 1, Hi, Wi].
        '''
        C = feats.shape[2]
        B, CAM_NUM, X, Ho, Wo  = grids.shape[0:5]

        cam_batched_feats = torch.flatten(feats, 0, 1)
        cam_batched_grids = torch.flatten(grids, 0, 1)

        vol = []

        for i in range(X):

            sampled_feat = grid_sample(
                cam_batched_feats,
                cam_batched_grids[:, i, ... ],
                align_corners=False
            )

            sampled_feat = sampled_feat.view(B, CAM_NUM, C, Ho, Wo) # (B, C#, C, H, W)
            sampled_feat = sampled_feat.flatten(1,2)

            sampled_feat = sampled_feat.unsqueeze(2)
            vol.append(sampled_feat)


        vol = torch.cat(vol, dim=2)
        return vol 

    def forward(self, feats:Tensor, grids:Tensor, grid_masks:Tensor, masks:Tensor):
        '''
        masks: should be in format (C#, Ho, Wo)
        feats: should be in format (B, C#, C, Hi, Wi)
        grids: should be in format (B, C#, X, Ho, Wo, 2), 
            where X is computed by number of inv dist candidates and 
            last dimension ∈ [-1,1].
        grid_masks: should be in format (B, C#, X, Ho, Wo, 1)
        Output Volume should be in format (B, X, (C#*C)/self.fuse_reduce, Ho, Wo)
        
        NOTE: grid_masks is not used. It is here for compatibility.
        
        '''

        # For each cameras' feature map, sweep for all X grids to get X tensors 
        # of size [ B, C, X, Hout, Wout ]
        # vol = torch.zeros(
        #     ( feats.shape[0],      # Batch
        #       self.feat_chs,       # Final channels in the cost volumne
        #       grids.shape[2],
        #       grids.shape[3],
        #       grids.shape[4]), # X x Ho x Wo
        #     dtype=feats.dtype,
        #     device=feats.device
        # )

        # Sweep
        vol = self.sweep(feats, grids, masks)

        # Post-process.
        vol = self.post_vol(vol)

        return vol