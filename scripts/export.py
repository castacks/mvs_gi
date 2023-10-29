# import torch_tensorrt
# torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Debug)

import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dsta_mvs.test.utils import *

import deform_conv2d_onnx_exporter
deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()


def main():
    model = load_model('wandb_logs/dsta-mvs-refactor/fp32-benchmark/checkpoints/epoch=99-step=100.ckpt')
    
    model.eval().cuda()

    dataloader = make_dataloader('/dataset/DSTA_MVS_Dataset_V2')
    sample = next(iter(dataloader))

    imgs        = sample['imgs'].cuda()
    grids       = sample['grids'].cuda()
    masks       = sample['masks'].cuda()

    # print('Exporting Torchscript...')
    
    # ts_module = torch.jit.trace(model, example_inputs = (imgs, grids, masks))
    # torch.jit.save(ts_module, 'artifacts/model.pt')

    # print('Exporting TRT...')

    # trt_ts_module = torch_tensorrt.compile(
    #     ts_module,
    #     inputs = [imgs, grids, masks],
    #     enabled_precisions = {torch.float32},
    #     # enabled_precisions = {torch.half}, # Run with FP16
    #     truncate_long_and_double=True,
    #     # require_full_compilation=True,
    #     # device=torch_tensorrt.Device("cuda:0", allow_gpu_fallback=True)
    # )

    # torch.jit.save(trt_ts_module, "model_trt.pt")

    print('Exporting ONNX...')
    
    torch.onnx.export(
        model,
        (imgs, grids, masks),
        'artifacts/model_backport.onnx',
        input_names = ['imgs', 'grids', 'masks'],
        output_names = ['inv_dist'],
        opset_version=13                # Jetpack 5.1 supports opset16
    )

    print('Done.')

if __name__ == '__main__':
    main()