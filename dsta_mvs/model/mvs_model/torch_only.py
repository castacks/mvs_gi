from torch import nn, Tensor
import torch.nn.functional as F

class SphericalSweepStereoBase(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        cv_builder: nn.Module,
        cv_regulator: nn.Module,
        dist_regressor: nn.Module
    ):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.cv_builder = cv_builder
        self.cv_regulator = cv_regulator
        self.dist_regressor = dist_regressor
        

    def extract_features(self, imgs:Tensor) -> Tensor:
        # Remember original shape and change format to ( B * C#, C, H, W )
        og_shp = imgs.shape[:2]
        imgs   = imgs.view( ( imgs.shape[0]*imgs.shape[1], *imgs.shape[2:] ) )

        feats = self.feature_extractor(imgs)
        feats = feats.view( ( *og_shp, *feats.shape[1:] ) )

        return feats
    
    def forward(self, imgs: Tensor, grids: Tensor, grid_masks: Tensor, masks: Tensor):
        feats = self.extract_features(imgs)
        vol = self.cv_builder(feats, grids, grid_masks, masks)
        costs = self.cv_regulator(vol)
        inv_dist = self.dist_regressor(costs)

        return inv_dist