from typing import Sequence

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple, Sequence


class VolumeCrossEntropy(nn.Module):
    def __init__(
        self,
        bf: float = 96,
        dist_list: Sequence[float]= [0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100],
    ):
        super().__init__()

        self.register_buffer('inv_dist_list', bf/torch.Tensor(dist_list), persistent=False) # N
        self.register_buffer('inv_dist_list_flipped', torch.flip(self.inv_dist_list, [0]), persistent=False)
        self.register_buffer('bin_widths', torch.diff(self.inv_dist_list), persistent=False) # N-1

    def forward(self, pred_cost: Tensor, inv_dist_true: Tensor, mask: Tensor = None) -> Tensor:
        inv_dist_true = torch.clamp(inv_dist_true, self.inv_dist_list[-1], self.inv_dist_list[0])

        # find bin indexes for true inv dist
        bin_idx = len(self.inv_dist_list) - torch.bucketize(inv_dist_true, self.inv_dist_list_flipped)
        bin_idx = torch.clamp(bin_idx-1, min=0, max=len(self.inv_dist_list)-2).to(torch.long) # Bx1xHxW
        bin_idx_flat = bin_idx.flatten()

        # index the lower bound distance candidates
        lb = self.inv_dist_list[bin_idx_flat].view_as(bin_idx) #Bx1xHxW
        
        # index the bin widths
        bin_widths = self.bin_widths[bin_idx_flat].view_as(bin_idx) # Bx1xHxW

        # calculate interpolation value
        d = (inv_dist_true - lb)/bin_widths # Bx1xHxW

        true_cost = torch.zeros_like(pred_cost) # BxNxHxW

        true_cost.scatter_(1, bin_idx, 1-d) # set lower bound weight
        true_cost.scatter_(1, bin_idx+1, d) # set upper bound weight

        if mask is not None:
            loss = F.binary_cross_entropy(pred_cost[mask], true_cost[mask], reduction='mean')
        else:
            loss = F.binary_cross_entropy(pred_cost, true_cost, reduction='mean')
        
        return loss
    

class MaskedSmoothL1Loss(nn.Module):

    def __init__(self, dist_regressor: nn.Module):

        super().__init__()
        self.loss = torch.nn.SmoothL1Loss()
        self.dist_regressor = dist_regressor

    def forward(self, pred_costs: Tensor, labels: Tensor, mask: Tensor = None):
        if pred_costs.shape[1] > 1:
            pred_costs = pred_costs.unsqueeze(1)
            pred_img = self.dist_regressor(pred_costs)
        else:
            pred_img = pred_costs

        if mask is None:
            return self.loss(pred_costs, labels)
        else:
            mask = mask[:,0,...].unsqueeze(1)
            return self.loss(pred_img[mask], labels[mask])
            

class InBoundsBCEDistanceLoss(nn.Module):

    def __init__(self, 
                 dist_regressor: nn.Module,
                 far_dist_threshold: float,
                 near_dist_threshold: float):
        
        super().__init__()
        self.loss = torch.nn.BCELoss(reduction='mean')
        self.dist_regressor = dist_regressor
        self.far_inv_dist_thresh = 1.0/far_dist_threshold
        self.near_inv_dist_thresh = 1.0/near_dist_threshold
        
    def forward(self, pred_costs: Tensor, labels: Tensor, mask: Tensor = None):
        if pred_costs.shape[1] > 1:
            pred_costs = pred_costs.unsqueeze(1)
            pred_img = self.dist_regressor(pred_costs)
            mask = mask[:,0,...].unsqueeze(1)
        else:
            pred_img = pred_costs

        pred_near = pred_img < self.near_inv_dist_thresh
        pred_far = pred_img > self.far_inv_dist_thresh
        pred_inbounds = torch.logical_and(pred_near, pred_far).float()

        label_near = labels < self.near_inv_dist_thresh
        label_far = labels > self.far_inv_dist_thresh
        label_inbounds = torch.logical_and(label_near, label_far).float()

        if mask is None:
            return self.loss(pred_inbounds, label_inbounds)
        else:
            return self.loss(pred_inbounds[mask], label_inbounds[mask])


class NearFocusDistanceLoss(nn.Module):

    def __init__(
        self,
        near_loss_func: nn.Module,
        far_loss_func: nn.Module,
        near_dist_thresh: float,
        loss_weights: Tuple[float, float],
        dist_list: Sequence
    ):
        super().__init__()

        self.near_loss_func = near_loss_func
        self.far_loss_func = far_loss_func
        self.inv_near_dist_thresh = 1.0 / near_dist_thresh
        self.near_weight = loss_weights[0]
        self.far_weight = loss_weights[1]
        self.num_cands = len(dist_list)
    
    def forward(self, pred_cost: Tensor, inv_dist_true: Tensor):

        far_mask = inv_dist_true < self.inv_near_dist_thresh
        far_vol_mask = far_mask.repeat(1,self.num_cands,1,1)

        far_loss = self.far_loss_func( pred_cost, inv_dist_true, mask = far_vol_mask )
        near_loss = self.near_loss_func( pred_cost, inv_dist_true, mask = ~far_vol_mask )

        loss = (self.near_weight * near_loss) + (self.far_weight * far_loss)

        return loss