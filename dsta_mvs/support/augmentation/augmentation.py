from typing import Callable, Tuple, Sequence

import torch
from torch import nn, Tensor

import kornia.augmentation as KA


def apply_to_multiview(fn: Callable, imgs: Tensor) -> Tensor:
    '''
    Apply transformation expecting (B,C,H,W) to multi-image tensor
    of shape (B,C#,C,H,W).
    '''
    original_shape = imgs.shape[:2]
    imgs = imgs.view((imgs.shape[0]*imgs.shape[1], *imgs.shape[2:])) # reshape to (B*C#, C, H, W)
    out = fn(imgs)
    out = out.view((*original_shape, *out.shape[1:])) # reshape output to match (B,C#,...)
    return out

class ImageAugmentation(nn.Module):
    def __init__(self, transform: Callable):
        super().__init__()
        self.transform = transform

    @torch.no_grad()
    def forward(self, batch:Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        imgs, grids, masks = batch
        imgs = apply_to_multiview(self.transform, imgs)
        return (imgs, grids, masks)
    
class ColorJiggle(ImageAugmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(KA.ColorJiggle(*args, **kwargs))

class RandomGaussianNoise(ImageAugmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(KA.RandomGaussianNoise(*args, **kwargs))

class RandomSingleImageMasking(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = KA.RandomErasing(*args, **kwargs)
        self.register_buffer('zero_pix', torch.Tensor([0.0]), persistent=False)

    @torch.no_grad()
    def forward(self, batch:Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        imgs, grids, masks = batch

        num_cameras = masks.shape[1]
        cam_idx = (torch.arange(masks.shape[0]), torch.randint(0, num_cameras, (masks.shape[0],)))
        
        selected_mask = masks[cam_idx] # (B, C, H, W)
        selected_mask = self.transform(selected_mask)
        masks[cam_idx] = selected_mask

        chosen_img = imgs[cam_idx]
        imgs[cam_idx] = torch.where(selected_mask.to(torch.bool), chosen_img, self.zero_pix)

        return imgs, grids, masks

class RandomSingleImageBlanking(nn.Module):
    @torch.no_grad()
    def forward(self, batch:Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        imgs, grids, masks = batch

        num_cameras = masks.shape[1]
        cam_idx = (torch.arange(masks.shape[0]), torch.randint(0, num_cameras, (masks.shape[0],)))

        masks[cam_idx] = torch.zeros_like(masks[cam_idx])
        imgs[cam_idx] = torch.zeros_like(imgs[cam_idx])

        return imgs, grids, masks

class AugmentationSequence(nn.Sequential):
    def __init__(self, transforms:Sequence[nn.Module]):
        super().__init__(*transforms)