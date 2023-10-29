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

def main(args):
    # dataloader = make_dataloader(
    #     '/dataset/DSTA_MVS_Dataset_V2', # TODO set dataset directory
    #     bf=96,
    #     dist_list=[0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100]
    # )

    # batch = next(iter(dataloader))

    # imgs        = batch['imgs'].numpy()
    # grids       = batch['grids'].numpy()
    # masks       = batch['masks'].numpy()

    sample = np.load('test_output/test_sample.npz')
    imgs = sample['imgs']
    grids = sample['grids']
    masks = sample['masks']

    model = onnx.load(args.model_dir)
    
    # sample = dict()

    # for input_ in model.graph.input:
    #     input_name = input_.name
    #     dim = input_.type.tensor_type.shape.dim
    #     input_shape = [int(MessageToDict(d).get("dimValue")) for d in dim]
    #     sample[input_name] = np.random.random(input_shape).astype(np.float32)

    # output_names = [output.name for output in model.graph.output]
    output_names = ['feats', 'vol', 'costs', 'inv_dist']

    session = ort.InferenceSession(
            args.model_dir,
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

    outputs = session.run(output_names, {'imgs': imgs, 'grids': grids, 'masks': masks})

    feats, vol, costs, inv_dist = outputs

    os.makedirs('test_output', exist_ok=True)
    np.savez(
        'test_output/test_output.npz',
        feats = feats,
        vol = vol,
        costs = costs,
        inv_dist = inv_dist,
    )

    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='model_dir', type=str)
    # parser.add_argument('-o', dest='out_dir', type=str, default='test_out.png')
    args = parser.parse_args()

    main(args)