import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..common import RELU_TYPE, NORM3D_TYPE
from ..common import common_modules as cm
from ..backports import bilinear_grid_sample as grid_sample

class SphericalSweepStdMasked(nn.Module):

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

    def sweep(self, feats:Tensor, grids:Tensor, grid_masks:Tensor, masks:Tensor):
        '''
        feats: [B, C#, C, Hi, Wi] .
        grids: [B, C#, X, Ho, Wo, 2], where X is the number of inv dist candidates.
        grid_masks: [B, C#, X, Ho, Wo, 1], where X is the number of inv dist candidates.
        masks: [B, C#, 1, Hi, Wi].
        '''
        C = feats.shape[2]
        B, CAM_NUM, X, Ho, Wo  = grids.shape[0:5]

        cam_batched_feats = torch.flatten(feats, 0, 1)
        cam_batched_masks = torch.flatten(masks, 0, 1)
        cam_batched_grids = torch.flatten(grids, 0, 1)

        # 2023-12-23 by Yaoyu: Need to convert bool to int to make it possible to convert ONNX 
        # opset 13 then TensorRT 8.2 for deployment on the Xavier NX with JetPack 4.6.1.
        # if cam_batched_masks.dtype == torch.bool:
        #     cam_batched_masks = cam_batched_masks.to(dtype=torch.float32)

        vol = []

        for i in range(X):
            grid_mask = grid_masks[:, :, i, :, :, 0] # [B, CAM_NUM, Ho, Wo]
            # Not useing unsqueeze to force the consistency between the features.
            grid_mask = grid_mask.view( B, CAM_NUM, 1, Ho, Wo )

            # Sample the features.
            # sampled_feat = F.grid_sample(
            #     cam_batched_feats,
            #     cam_batched_grids[:, i, ... ],
            #     mode = 'bilinear',
            #     padding_mode='zeros',
            #     align_corners=False
            # )

            sampled_feat = grid_sample(
                cam_batched_feats,
                cam_batched_grids[:, i, ... ],
                align_corners=False
            )

            sampled_feat = sampled_feat.view(B, CAM_NUM, C, Ho, Wo) # (B, C#, C, H, W)

            # Sample the masks.
            # sampled_mask = F.grid_sample(
            #     cam_batched_masks,
            #     cam_batched_grids[:, i, ... ],
            #     mode='nearest',
            #     padding_mode='zeros',
            #     align_corners=False
            # ).to(torch.bool)

            # Backport for Jetpack 4.6 support
            # TODO: This should happen outside the model.
            sampled_mask = grid_sample(
                cam_batched_masks,
                cam_batched_grids[:, i, ... ],
                align_corners=False
            ) > 0.0

            # Need to make sure mask has only 0 and 1s.
            sampled_mask = sampled_mask.view(B, CAM_NUM, 1, Ho, Wo)

            # Merge sampled_mask and the grid_mask.
            sampled_mask = torch.logical_and( sampled_mask, grid_mask )

            # Compute the number of valid features or every feature location.
            # (B, C#, 1, H, W)
            summed_mask = torch.sum( sampled_mask.to(torch.float), dim=1, keepdim=True )
            # valid if more than 1 camera is valid # (B, C#, 1, H, W)
            valid_output_mask = summed_mask > 1.0
            
            # (B, C#, 1, H, W)
            cam_count = torch.where(valid_output_mask, summed_mask, torch.ones(valid_output_mask.shape, dtype=summed_mask.dtype, device=valid_output_mask.device))

            # Compute the avg. (B, 1, C, H, W).
            avg_feat  = torch.sum( sampled_feat * sampled_mask.to(torch.float), dim=1, keepdim=True ) / cam_count

            # Override sampled_feat with avg at invalid output locations.
            # Such that sampled_feat - avg_feat becomes zero at invalid output locations, so std 
            # dev calculated is 0.
            sampled_feat = torch.where( sampled_mask, sampled_feat, avg_feat ) # (B, C#, C, H, W)

            # Compute the std. dev. (B, 1, C, H, W).
            std2_feat = torch.sum( (sampled_feat - avg_feat)**2, dim=1, keepdim=True ) / cam_count

            # zero out invalid output. (B, 1, C, H, W).
            std2_feat = torch.where(valid_output_mask.expand_as(std2_feat), std2_feat, torch.zeros(std2_feat.shape, dtype=std2_feat.dtype, device=valid_output_mask.device))

            # Squeeze causes problems with export: https://github.com/pytorch/pytorch/issues/79117
            # std2_feat = torch.squeeze(std2_feat, dim=1) # (B, C, H, W)
            std2_feat = std2_feat[:, 0, :, :, :]
            # vol[:,:,i,...] = std2_feat # copy to vol

            std2_feat = std2_feat.unsqueeze(2)
            vol.append(std2_feat)

        vol = torch.cat(vol, dim=2)
        return vol 

    def forward(self, feats:Tensor, grids:Tensor, grid_masks:Tensor, masks:Tensor):
        '''
        masks: should be in format (C#, Ho, Wo)
        feats: should be in format (B, C#, C, Hi, Wi)
        grids: should be in format (B, C#, X, Ho, Wo, 2), 
            where X is computed by number of inv dist candidates and 
            last dimension ∈ [-1,1].
        grid_masks: (B, C#, X, Ho, Wo, 1), valid mask of every sweep of grid.
        Output Volume should be in format (B, X, (C#*C)/self.fuse_reduce, Ho, Wo)
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
        vol = self.sweep(feats, grids, grid_masks, masks)

        # Post-process.
        vol = self.post_vol(vol)

        return vol