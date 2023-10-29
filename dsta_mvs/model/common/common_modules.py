import copy
import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d

from .utils import calc_shape_conv2d

class NoOp(nn.Identity):
    '''Alias for `nn.Identity` to support shape inference.'''
    def infer_size(self, in_size:Tuple[int]) -> Tuple[int, int]:
        return in_size        

class BaseConvBlk2d(nn.Module):
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int,
        kernel_size: int,
        stride: int = 1,
        out_pad: int = 0,
        extra_pad: int = 0,
        bias_on: bool = False, 
        norm_layer: nn.Module = NoOp(), 
        activation: nn.Module = NoOp()
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.out_pad = out_pad
        self.extra_pad = extra_pad

        self.conv_layer = nn.Conv2d(
                            in_chs,
                            out_chs,
                            kernel_size,
                            padding=(kernel_size//2)+extra_pad, 
                            bias=bias_on,
                            stride=stride
                        )
        
        self.norm_layer = norm_layer
        self.activation = activation

        self.out_pad = out_pad
        
        if out_pad > 0:
            self.pad_layer = nn.ZeroPad2d((self.out_pad, self.out_pad, self.out_pad, self.out_pad))
        else:
            self.pad_layer = NoOp()

    def forward(self, x: Tensor, res: Optional[Tensor] = None):
        x = self.conv_layer(x)
        x = self.norm_layer(x)

        if res is not None:
            x = x + res

        x = self.activation(x)

        x = self.pad_layer(x)

        return x
    
    def infer_size(self, in_size:Tuple[int]) -> Tuple[int, int]:
        '''Get output size.'''
         
        H, W = calc_shape_conv2d(
            in_size = in_size,
            kernel_size = self.kernel_size,
            stride = self.stride,
            padding = (self.kernel_size//2) + self.extra_pad
        )

        return (H+2*self.out_pad, W+2*self.out_pad)
    
class BaseConvBlk3d(nn.Module):
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int,
        kernel_size: int,
        stride: int = 1,
        extra_pad: int = 0,
        bias_on: bool = False, 
        norm_layer: nn.Module = NoOp(), 
        activation: nn.Module = NoOp()
    ):
        super().__init__()

        
        self.conv_layer = nn.Conv3d(in_chs, out_chs, 
                            kernel_size, padding=(kernel_size//2)+extra_pad, 
                            bias=bias_on,
                            stride=stride
                        )

        self.norm_layer = norm_layer
        self.activation = activation


    def forward(self, x: Tensor, res: Optional[Tensor] = None):
        x = self.conv_layer(x)
        x = self.norm_layer(x)

        if res is not None:
            x = x + res
        x = self.activation(x)

        return x
    
class ResConvBlk2d(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int = 3,
        in_stride: int = 1,
        out_stride: int = 1,
        out_pad: int = 0,
        activation: nn.Module = NoOp(),
        norm_layer: nn.Module = NoOp(),
    ):
        super().__init__()

        self.in_chs     = in_chs
        self.out_chs    = out_chs
        self.k_sz       = kernel_size
        self.in_stride  = in_stride
        self.out_stride = out_stride
        self.out_pad    = out_pad

        self.blk1 = BaseConvBlk2d(
            self.in_chs, self.out_chs, self.k_sz, 
            stride=self.in_stride,
            activation=copy.deepcopy(activation),
            norm_layer=copy.deepcopy(norm_layer)
        )

        self.blk2 = BaseConvBlk2d(
            self.out_chs, self.out_chs, self.k_sz, 
            stride=self.out_stride, 
            activation=copy.deepcopy(activation),
            norm_layer=copy.deepcopy(norm_layer)
        )

        if self.in_chs != self.out_chs:
            self.one_by_one = BaseConvBlk2d(
                self.in_chs, self.out_chs, 1, 
                stride=self.out_stride*self.in_stride,
                activation=copy.deepcopy(activation), 
                norm_layer=copy.deepcopy(norm_layer),
            )
        else:
            self.one_by_one = NoOp()

        if out_pad > 0:
            self.pad_layer = nn.ZeroPad2d((self.out_pad, self.out_pad, self.out_pad, self.out_pad))
        else:
            self.pad_layer = NoOp()

    def forward(self, x: Tensor):
        res = self.blk1(x)

        x = self.one_by_one(x)

        res = self.blk2(res,res=x)

        res = self.pad_layer(res)
        
        return res
    
    def infer_size(self, in_size:Tuple[int]) -> Tuple[int, int]:
        '''Get output size.'''
        size = self.blk1.infer_size(in_size)
        size = self.one_by_one.infer_size(size)
        size = self.blk2.infer_size(size)
        size = (size[0] + 2*self.out_pad, size[1] + 2*self.out_pad)
        return size
    
class ResConvBlk3d(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int = 3,
        in_stride: int = 1,
        out_stride: int = 1,
        out_pad: int = 0,
        activation: nn.Module = NoOp(),
        norm_layer: nn.Module = NoOp(),
    ):
        super().__init__()

        self.in_chs     = in_chs
        self.out_chs    = out_chs
        self.k_sz       = kernel_size
        self.in_stride  = in_stride
        self.out_stride = out_stride
        self.out_pad    = out_pad

        self.blk1 = BaseConvBlk3d(
            self.in_chs, self.out_chs, self.k_sz, 
            stride=self.in_stride,
            activation=copy.deepcopy(activation),
            norm_layer=copy.deepcopy(norm_layer)
        )

        self.blk2 = BaseConvBlk3d(
            self.out_chs, self.out_chs, self.k_sz, 
            stride=self.out_stride, 
            activation=copy.deepcopy(activation),
            norm_layer=copy.deepcopy(norm_layer)
        )

        if self.in_chs != self.out_chs:
            self.one_by_one = BaseConvBlk3d(
                self.in_chs, self.out_chs, 1, 
                stride=self.out_stride*self.in_stride,
                activation=copy.deepcopy(activation), 
                norm_layer=copy.deepcopy(norm_layer),
            )
        else:
            self.one_by_one = NoOp()

    def forward(self, x: Tensor):
        res = self.blk1(x)

        x = self.one_by_one(x)

        res = self.blk2(res,res=x)

        # NOTE: Different padding setting with BaseConvBlk.
        res = F.pad(res,
            (self.out_pad, self.out_pad, # Left, right.
                self.out_pad, self.out_pad) # Top, bottom.
        )
        
        return res

class PixelUnshuffle(nn.PixelUnshuffle):
    def forward(self, x: Tensor):
        return super()(x.permute([0,2,1,3,4])).permute([0,2,1,3,4])
    
class PixelShuffle(nn.PixelShuffle):
    def forward(self, x: Tensor):
        return super()(x.permute([0,2,1,3,4])).permute([0,2,1,3,4])

class ResizeConv2d(nn.Module):
    def __init__(
        self, 
        in_chs: int,
        out_chs: int,
        kernel_size: int, 
        stride: int = 1, 
        extra_pad: int = 0, 
        out_pad: int = 0,
        activation: Tensor = NoOp(), 
        norm_layer: Tensor = NoOp(),
    ):
        super().__init__()

        self.scale         = stride
        self.out_pad       = out_pad
        self.conv = BaseConvBlk2d(
            in_chs,
            out_chs,
            kernel_size, 
            stride=1,
            extra_pad=extra_pad,
            activation=copy.deepcopy(activation),
            norm_layer=copy.deepcopy(norm_layer)
        )

    def forward(self, x: Tensor, res: Optional[Tensor] = None):
        up_sz = [ int(self.scale * s) + self.out_pad for s in x.shape[2:] ]
        
        x = F.interpolate(
                x,
                size=up_sz,
                scale_factor=None,
                mode='bilinear',
                align_corners=False
            )
        
        if res is not None:
            if x.shape != res.shape:
                # NOTE: Was not using self.corners.
                x = F.interpolate(
                    x,
                    size=res.shape[2:],
                    mode='bilinear'
                )
            x = self.conv(x, res=res)
        else:
            x = self.conv(x)

        return x
    
class ResizeConv3d(nn.Module):
    def __init__(
        self, 
        in_chs: int,
        out_chs: int,
        kernel_size: int, 
        stride: int = 1, 
        extra_pad: int = 0, 
        out_pad: int = 0,
        activation: Tensor = NoOp(), 
        norm_layer: Tensor = NoOp(),
    ):
        super().__init__()

        self.scale         = stride
        self.out_pad       = out_pad

        self.conv = BaseConvBlk3d(
            in_chs,
            out_chs,
            kernel_size, 
            stride=1,
            extra_pad=extra_pad,
            activation=copy.deepcopy(activation),
            norm_layer=copy.deepcopy(norm_layer)
        )

    def forward(self, x: Tensor, res: Optional[Tensor] = None):
        up_sz = [ int(self.scale * s) + self.out_pad for s in x.shape[2:] ]
        
        x = F.interpolate(
                x,
                size=up_sz,
                scale_factor=None,
                mode='trilinear',
                align_corners=False
            )
        
        if res is not None:
            if x.shape != res.shape:
                # NOTE: Was not using self.corners.
                x = F.interpolate(
                    x,
                    size=res.shape[2:],
                    mode='trilinear'
                )
            x = self.conv(x, res=res)
        else:
            x = self.conv(x)

        return x
    
# Adpated from https://github.com/nju-ee/MODE-2022/blob/main/models/basic/spherical_conv/sphere_conv.py
# deform_conv2d not supported by ONNX (might be available in opset >=19)
# Torchscript does not like subclassing (https://github.com/pytorch/pytorch/issues/42885)
class SphereConvEquirect2d(nn.Module):
    def __init__(
        self,
        in_size: Tuple[int, int],
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()
        
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1])
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        # pre-compute offset
        self.register_buffer('offset', self.gen_offset(in_size, self.kernel_size, self.stride, self.padding, self.dilation))

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, mask: Optional[Tensor] = None):
        B = input.size()[0]
        offset = self.offset.expand([B, -1, -1, -1])

        return deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )

    @staticmethod
    def gen_offset(input_size, kernel_size, stride, padding, dilation, lat_range=(-math.pi/2,0), lon_range=(0, 2*math.pi)):
        height, width = input_size[-2:]
        Kh, Kw = kernel_size
        stride_h, stride_w = stride
        dilation_h, dilation_w = dilation
        
        lat_dist = lat_range[1] - lat_range[0]
        lon_dist = lon_range[1] - lon_range[0]

        lat_center = lat_dist/2 + lat_range[0]
        lon_center = lon_dist/2 + lon_range[0]

        delta_lat = lat_dist / height
        delta_lon = lon_dist / width

        range_y = torch.arange(-(Kh // 2), Kh // 2 + 1)
        range_x = torch.arange(-(Kw // 2), Kw // 2 + 1)
        
        if not Kw % 2:
            # range_x = np.delete(range_x, Kw // 2)
            range_x = range_x[:-1]
        
        if not Kh % 2:
            # range_y = np.delete(range_y, Kh // 2)
            range_y = range_y[:-1]

        kerX = torch.tan(range_x * dilation_w * delta_lon)
        kerY = torch.tan(range_y * dilation_h * delta_lat) / torch.cos(range_y * delta_lon)
        
        # Note: Torch and NumPy meshgrid behaviour is different.
        # Torch may implement NumPy behaviour is future.
        # Original: kerX, kerY = np.meshgrid(kerX, kerY)
        kerY, kerX = torch.meshgrid(kerY, kerX)

        rho = torch.sqrt(kerX**2 + kerY**2)
        # when the value of rho at center is zero, some lat values explode to `nan`.
        if Kh % 2 and Kw % 2:
            rho[Kh // 2][Kw // 2] = 1e-8

        nu = torch.arctan(rho)
        cos_nu = torch.cos(nu)
        sin_nu = torch.sin(nu)
        h_range = torch.arange(0, height, stride_h) + 0.5
        w_range = torch.arange(0, width, stride_w) + 0.5

        lat_range = ((h_range / height) - 0.5) * lat_dist + lat_center
        lon_range = ((w_range / width) - 0.5) * lon_dist + lon_center
        
        # generate sampling pattern
        _lat = lat_range[0]
        lat = torch.stack([torch.arcsin(cos_nu * torch.sin(_lat) + kerY * sin_nu * torch.cos(_lat) / rho) for _lat in lat_range])  # (H, Kh, Kw)

        lat = torch.stack([lat for _ in lon_range])  # (W, H, Kh, Kw)
        lat = lat.permute((1, 0, 2, 3))  # (H, W, Kh, Kw)

        # generate longitude sampling pattern
        # Note: use atan2 for 2pi value range
        lon = torch.stack([torch.arctan2(kerX * sin_nu, (rho * torch.cos(_lat) * cos_nu - kerY * torch.sin(_lat) * sin_nu)) for _lat in lat_range])  # (H, Kh, Kw)

        lon = torch.stack([lon + _lon for _lon in lon_range])  # (W, H, Kh, Kw)
        lon = lon.permute((1, 0, 2, 3))  # (H, W, Kh, Kw)

        # convert lat lon to pixel index
        lat = ((lat - lat_center) / lat_dist + 0.5) * height  # (H, W, Kh, Kw)
        lon = ((lon - lon_center) / lon_dist + 0.5) * width   # (H, W, Kh, Kw)

        # wraparound longitude
        lon = lon % (width - 1)

        lon = lon % width # wraparound longitude

        # convert to offset
        H, W = torch.meshgrid(h_range, w_range)
        lat = lat - (H[..., None, None] - 0.5)
        lon = lon - (W[..., None, None] - 0.5)

        LatLon = torch.stack((lat, lon)).to(torch.float)  # (2, H, W, Kh, Kw) = ((lat, lon), H, W, Kh, Kw)
        LatLon = LatLon.permute((3, 4, 0, 1, 2))  # (Kh, Kw,2, H, W) = (Kh, Kw,(lat, lon), H, W)
        Kh, Kw, d, H, W = LatLon.shape
        LatLon = LatLon.reshape((1, d * Kh * Kw, H, W))  # (1, 2*Kh*Kw, H, W)

        return LatLon

class SphereConvBlk(nn.Module):
    def __init__(
        self,
        in_size: Tuple[int, int],
        in_chs: int,
        out_chs: int,
        k_sz: int,
        stride: int = 1,
        extra_pad:int = 0,
        bias: bool = False,
        dilation:int = 1,
        norm_layer: nn.Module = NoOp(),
        activation: nn.Module = NoOp()
    ):
        super().__init__()

        blk = []
        blk.append(SphereConvEquirect2d(
            in_size = in_size,
            in_channels = in_chs,
            out_channels = out_chs,
            kernel_size = k_sz,
            stride = stride,
            padding = k_sz//2 + extra_pad,
            dilation = dilation,
            bias = bias
        ))
        
        blk.append(norm_layer)

        self.activation = activation

        self.blk = nn.Sequential(*blk)

    def forward(self, x: Tensor, res: Optional[Tensor] = None):

        x = self.blk(x)
        if res is not None:
            x = x + res
        
        x = self.activation(x)

        return x