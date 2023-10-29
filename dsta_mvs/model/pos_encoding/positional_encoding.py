# PyTorch packages.
import torch.nn as nn
import torch
from typing import Callable
import math

from typing import Sequence, Tuple, Type


class PositionalEncoder(nn.Module):
    def __init__(self,
        c_dim: int,
        coord_dims: Tuple[int, int],
        coord_encoder: nn.Module
        ):

        super().__init__()

        self.c_dim = c_dim
        self.coord_dims = coord_dims
        self.coord_encoder = coord_encoder
        self.out_dim = self.coord_encoder.out_dim
    

    def forward(self, x):
        x_shape = x.shape
        coord_rngs = [torch.arange(x_shape[i], device=x.device) for i in self.coord_dims]
        coord_grids = [i.unsqueeze(-1) for i in torch.meshgrid(*coord_rngs)]
        coord_grids = torch.cat(coord_grids,dim=-1).unsqueeze(0)
        coord_grids = coord_grids.repeat(x.shape[0],1,1,1)

        pos_emb = self.coord_encoder(coord_grids).permute(0,3,1,2)
        x_emb = torch.cat([pos_emb, x], dim=self.c_dim)

        return x_emb


class SinusoidalEncoding(nn.Module):
    def __init__(self, num_freqs):
        super().__init__()

        self.out_dim = 2 * num_freqs
        
        self.inv_freq_nums = 1.0 / (10000 ** (torch.arange(0, num_freqs).float() / num_freqs))
        self.register_buffer("inv_freq", self.inv_freq_nums)

    def forward(self, x):
        x_sin = torch.sin(x[..., None] * self.inv_freq_nums.to(x.device))
        x_cos = torch.cos(x[..., None] * self.inv_freq_nums.to(x.device))

        return torch.cat([x_sin, x_cos], dim=-1).flatten(-2, -1)