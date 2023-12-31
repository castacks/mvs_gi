# import torch_tensorrt
# torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Debug)

import os
import sys

import torch
from lightning import Trainer, LightningModule

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dsta_mvs.test.utils import *

import deform_conv2d_onnx_exporter
deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()

def main():

    model_in = 'code_release_202310_wd/CR_EV004/dsta_sweep_config103_WB_jy2dqg6r_v102.ckpt'
    dir_out = 'code_release_202310_trt/artifacts_op13_gi/config103'
    opset_version = 13 # Jetpack 5.1 supports opset16

    if not os.path.exists(dir_out):
        os.makedirs(dir_out, exist_ok=True)

    model = load_model(model_in)
    # import ipdb; ipdb.set_trace()
    model.eval().cuda()

    dataloader = make_dataloader(
        'code_release_202310_data/synthetic',
        dist_list=[0.5, 0.5436655530390901, 0.5933598538089077, 0.6505899686755604, 0.7174058117693126, 0.7966693499997578, 0.892501608318949, 1.0110614299826677, 1.161983802830304, 1.3612365930547528, 1.637348559910646, 2.0467867239341255, 2.7192485658566654, 4.033339718346582, 7.764432455310304, 100.0], 
        # csv_rig_rgb_suffix='_rgb_equirect',
        # csv_rig_dist_suffix='_dist_equirect',
        csv_rig_rgb_suffix='_rgb_fisheye', 
        csv_rig_dist_suffix='_dist_fisheye',
        )
    sample = next(iter(dataloader))

    imgs       = sample['imgs']
    grids      = sample['grids']
    grid_masks = sample['grid_masks']
    masks      = sample['masks']

    # 2023-12-23 by Yaoyu: Need to convert bool to int to make it possible to convert ONNX 
    # opset 13 then TensorRT 8.2 for deployment on the Xavier NX with JetPack 4.6.1.
    grid_masks = grid_masks.to(dtype=torch.float32)
    masks      = masks.to(dtype=torch.float32)

    # Save the tensors as numpy arrays to a .npz file for testing purpose.
    np.savez( os.path.join( dir_out, 'test_sample.npz' ), 
             imgs=imgs.numpy(), 
             grids=grids.numpy(), 
             grid_masks=grid_masks.numpy(),
             masks=masks.numpy() )
    
    # Push the tenors to GPU.
    imgs       = imgs.cuda()
    grids      = grids.cuda()
    grid_masks = grid_masks.cuda()
    masks      = masks.cuda()

    print('Exporting ONNX...')
    
    torch.onnx.export(
        model,
        (imgs, grids, grid_masks, masks),
        os.path.join( dir_out, 'model_backport.onnx' ),
        input_names = ['imgs', 'grids', 'grid_masks', 'masks'],
        output_names = ['inv_dist'],
        opset_version=opset_version
    )

    print('Done.')

if __name__ == '__main__':
    main()