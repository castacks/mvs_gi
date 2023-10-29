from typing import Sequence, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F

class DistanceRegressorWithFixedCandidates(nn.Module):
    def __init__(
        self,
        bf: float = 96,
        dist_cands: Sequence[float] = [0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100],
        interp_scale_factor: float = 0, # formerly `out_interp`
        pre_interp: bool = False,
    ):
        '''
        Parameters
        ----------
        dist_cands
            List of distance candidates.
        bf
            Dummy baseline x focal length.
        interp_scale_factor
            The interpolation/resize factor for the output.
        pre_interp
            True if interpolation should be done before softmax.
        '''
        super().__init__()

        inv_dist_idx = bf / torch.tensor(dist_cands, dtype=torch.float32)
        # This is a bug. The condidates should not be module parameters.
        # self.inv_dist_idx = torch.from_numpy(inv_dist_idx).view(1, -1, 1, 1).cuda()
        self.register_buffer( 'inv_dist_idx', inv_dist_idx.view(1, -1, 1, 1), persistent=True )
        self.inv_dist_idx_min = float( torch.min(self.inv_dist_idx) )
        self.inv_dist_idx_max = float( torch.max(self.inv_dist_idx) )
        self.bf = bf

        self.interp_scale_factor = interp_scale_factor if interp_scale_factor > 0 else 0
        self.pre_interp = pre_interp

    def update_dist_cands(self, dist_cands: Sequence[float]):
        inv_dist_idx = self.bf / torch.tensor(dist_cands, dtype=torch.float32)
        # Not sure if this is the best way to update a buffer.
        # https://stackoverflow.com/questions/67909883/updating-a-register-buffer-in-pytorch
        self.inv_dist_idx[:] = inv_dist_idx.view(1, -1, 1, 1).to(
            dtype=self.inv_dist_idx.dtype,
            device=self.inv_dist_idx.device )
        
        self.inv_dist_idx_min = float( torch.min(self.inv_dist_idx) )
        self.inv_dist_idx_max = float( torch.max(self.inv_dist_idx) )

    def forward(self, costs: Tensor) -> Tensor:
        distance_cost = costs[:, 0, ...]
        
        if ( self.pre_interp and self.interp_scale_factor > 0 ):
            distance_cost = F.interpolate(
                distance_cost, 
                size=None,
                scale_factor=self.interp_scale_factor,
                mode='bilinear'
            )

        norm_costs = F.softmax(distance_cost, 1)
        
        if ( not self.pre_interp and self.interp_scale_factor > 0 ):
            norm_costs = F.interpolate(
                norm_costs, 
                size=None,
                scale_factor=self.interp_scale_factor,
                mode='nearest'
            )

        inv_depths = torch.sum(
            norm_costs * self.inv_dist_idx.expand_as(norm_costs),
            dim=1,
            keepdim=True
        )

        return inv_depths