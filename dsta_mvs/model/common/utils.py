from typing import Tuple

def calc_shape_conv2d(
        in_size: Tuple[int, int],
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1
    ) -> Tuple[int, int]:
    '''Calculate the output spatial dimensions of Conv2d.'''
    H, W = in_size

    Ho = (H + 2*padding - dilation * (kernel_size-1) - 1)/stride + 1
    Wo = (W + 2*padding - dilation * (kernel_size-1) - 1)/stride + 1

    return (int(Ho), int(Wo))