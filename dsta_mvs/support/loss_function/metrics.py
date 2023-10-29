from typing import Sequence, Tuple

import torch
from torch import nn, Tensor
# import torch.nn.functional as F

from torchmetrics.functional import (
    mean_absolute_error,
    mean_squared_error,
    structural_similarity_index_measure )

DEFAULT_BF = 96
DEFAULT_DIST_LIST = [0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100]

class MVSMetric(nn.Module):
    def __init__(
            self,
            bf: float=DEFAULT_BF,
            dist_list: Sequence[float]=DEFAULT_DIST_LIST,
        ):
        super().__init__()
        self.bf = bf
        inv_dist_list = bf / torch.Tensor(dist_list)
        self.register_buffer('clamp_min', torch.min(inv_dist_list), persistent=False)
        self.register_buffer('clamp_max', torch.max(inv_dist_list), persistent=False)

    def clamp_and_scale(self, 
                        preds: Tensor,
                        target: Tensor
        ) -> Tuple[Tensor, Tensor]:

        # Clamp true values to dist candidate range? Convert to m^-1.
        target = torch.clamp(target, self.clamp_min, self.clamp_max)
        target = target / self.bf
        preds  = preds  / self.bf

        return preds, target

    def forward(self, 
                preds: Tensor, 
                target: Tensor, 
                valid_mask: Tensor=None 
        ) -> Tensor:
        raise NotImplementedError()

class SSIMMetric(MVSMetric):
    def __init__(
        self,
        bf: float=DEFAULT_BF,
        dist_list: Sequence[float]=DEFAULT_DIST_LIST,
        ):
        super().__init__(bf=bf, dist_list=dist_list)

    def forward(self, 
                preds: Tensor, 
                target: Tensor, 
                valid_mask: Tensor=None
        ) -> Tensor:
        '''valid_mask cannot be used. It is here for compatibility with other metrics.
        '''

        preds, target = self.clamp_and_scale(preds, target)

        loss = structural_similarity_index_measure(
            preds,
            target )

        return loss
    
class RMSEMetric(MVSMetric):
    def __init__(
        self,
        bf: float=DEFAULT_BF,
        dist_list: Sequence[float]=DEFAULT_DIST_LIST,
        ):
        super().__init__(bf=bf, dist_list=dist_list)

    def forward(self, 
                preds: Tensor, 
                target: Tensor, 
                valid_mask: Tensor=None
        ) -> Tensor:

        preds, target = self.clamp_and_scale(preds, target)

        if valid_mask is not None:
            preds  = preds[  valid_mask ]
            target = target[ valid_mask ]

        loss = mean_squared_error(
            preds,
            target,
            squared=False )

        return loss
    
class MAEMetric(MVSMetric):
    def __init__(
        self,
        bf: float=DEFAULT_BF,
        dist_list: Sequence[float]=DEFAULT_DIST_LIST,
        ):
        super().__init__(bf=bf, dist_list=dist_list)

    def forward(self, 
                preds: Tensor, 
                target: Tensor, 
                valid_mask: Tensor=None
        ) -> Tensor:
        
        preds, target = self.clamp_and_scale(preds, target)

        if valid_mask is not None:
            preds  = preds[  valid_mask ]
            target = target[ valid_mask ]

        loss = mean_absolute_error(
            preds,
            target )

        return loss
    
class BadPixelRatioMetric(MVSMetric):
    def __init__(
        self,
        bf: float=DEFAULT_BF,
        dist_list: Sequence[float]=DEFAULT_DIST_LIST,
        delta_thresh: float=0.1
        ):
        super().__init__(bf=bf, dist_list=dist_list)
        
        self.delta_thresh = delta_thresh

    def forward(self, 
                preds: Tensor, 
                target: Tensor, 
                valid_mask: Tensor=None
        ) -> Tensor:
        
        preds, target = self.clamp_and_scale(preds, target)

        #TODO: Need to debug the shape.
        delta = torch.abs( preds - target )

        if valid_mask is not None:
            delta = delta[valid_mask]
            n_elems = torch.sum(valid_mask)
        else:
            # TODO: This might not be correct since torch.sum() will reduce all dimensions of
            # the input tensor. Then the toal number of pixels is B*H*W, not H*W.
            H, W = delta.shape[-2:]
            n_elems = H * W

        bad_pix_ratio = torch.sum( delta > self.delta_thresh ) / n_elems \
                        if n_elems > 0 \
                        else 1.0 # TODO: Is this the best way to handle this?

        return bad_pix_ratio
    
class InverseMetricWrapper(nn.Module):
    def __init__(
        self,
        metric: nn.Module
    ):
        super().__init__()
        self.metric = metric

    def forward(self, 
                preds: Tensor, 
                target: Tensor, 
                valid_mask: Tensor=None
        ) -> Tensor:
       
        try:
            metric_score = self.metric( 
                            1.0 / preds, 
                            1.0 / target, 
                            valid_mask )
        except:
            # TODO: Is this the best way to handle this?
            metric_score = 0.0
        
        return metric_score
    
class LatitudeCropMetricWrapper(nn.Module):
    def __init__(
        self,
        metric: nn.Module,
        lat_rng: Tuple[float, float]      
    ):
        super().__init__()
        self.metric = metric
        self.lat_rng = lat_rng

    def forward(self, preds: Tensor, target: Tensor, valid_mask: Tensor=None) -> Tensor:
        # TODO: This function is not finished! Do not use.
        assert False, "LatitudeCropMetricWrapper is not finished! Do not use. "
        # TODO: need to debug the shape.
        B, C, H, W = preds.shape
        v_min, v_max = [H*(1-torch.sin(lat)) for lat in self.lat_rng]

        pred_cropped = preds[:, :, v_min:v_max, ...]
        target_cropped = target[:, :, v_min:v_max, ...]

        return self.metric(pred_cropped, target_cropped)