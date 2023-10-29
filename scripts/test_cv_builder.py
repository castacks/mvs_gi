import argparse
import time
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
import onnx
from google.protobuf.json_format import MessageToDict

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dsta_mvs.test.utils import *

def main():
    dataloader = make_dataloader(
        '/dataset/DSTA_MVS_Dataset_V2', # TODO set dataset directory
        bf=96,
        dist_list=[0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100]
    )

    batch = next(iter(dataloader))

    imgs        = batch['imgs'].numpy()
    grids       = batch['grids'].numpy()
    masks       = batch['masks'].numpy()

    np.savez(
        'test_output/test_sample.npz',
        imgs = imgs,
        grids = grids,
        masks = masks,
    )

    feats = np.load('test_output/feats.npy')

    model = onnx.load('artifacts/components/isolated_cv_builder_rn.onnx')
    
    output_names = [output.name for output in model.graph.output]


    session = ort.InferenceSession(
            'artifacts/components/isolated_cv_builder_rn.onnx',
            providers=[
                # ('TensorrtExecutionProvider', {
                #     # 'trt_engine_cache_enable': True,
                #     # 'trt_engine_cache_path': '/workspace/trt_cache',
                #     # 'trt_dla_enable': True,
                #     # 'trt_context_memory_sharing_enable': True,
                #     # 'trt_fp16_enable': True,
                #     # 'trt_int8_enable': True,
                # }),
                ('CUDAExecutionProvider', {
                    "cudnn_conv_use_max_workspace": True,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE', # EXHAUSTIVE good for static shapes
                    # 'do_copy_in_default_stream': False,
                }),
            ])

    outputs = session.run(output_names, {'feats': feats, 'grids': grids, 'masks': masks})

    output_dict = {k: v for k, v in zip(output_names, outputs)}

    os.makedirs('test_output', exist_ok=True)
    np.savez(
        'test_output/isolated_builder_output.npz',
        **output_dict
    )

    print('Done.')

if __name__ == '__main__':
    main()