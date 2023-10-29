import argparse
import time
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
import onnx

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dsta_mvs.test.utils import *

def main():
    vol = np.load('test_output/vol_pc.npy')

    model = onnx.load('artifacts/components/isolated_cv_regulator.onnx')
    
    # sample = dict()

    # for input_ in model.graph.input:
    #     input_name = input_.name
    #     dim = input_.type.tensor_type.shape.dim
    #     input_shape = [int(MessageToDict(d).get("dimValue")) for d in dim]
    #     sample[input_name] = np.random.random(input_shape).astype(np.float32)

    output_names = [output.name for output in model.graph.output]

    session = ort.InferenceSession(
            'artifacts/components/isolated_cv_regulator.onnx',
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

    outputs = session.run(output_names, {'vol': vol})

    os.makedirs('test_output', exist_ok=True)
    np.save(
        'test_output/isolated_output.npy',
        outputs
    )

    print('Done.')

if __name__ == '__main__':
    main()