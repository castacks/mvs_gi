import time
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dsta_mvs.test.utils import *

def main():
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

    session = ort.InferenceSession(
            'artifacts/model_backport.onnx',
            providers=[
                # ('TensorrtExecutionProvider', {
                #     'trt_engine_cache_enable': True,
                #     'trt_engine_cache_path': '/workspace/trt_cache',
                #     # 'trt_dla_enable': True,
                #     # 'trt_context_memory_sharing_enable': True,
                #     'trt_fp16_enable': True,
                #     # 'trt_int8_enable': True,
                # }),
                ('CUDAExecutionProvider', {
                    "cudnn_conv_use_max_workspace": True,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE', # EXHAUSTIVE good for static shapes
                    # 'do_copy_in_default_stream': False,
                }),
                # 'CPUExecutionProvider'
            ])

    # IO bind
    imgs = ort.OrtValue.ortvalue_from_numpy(imgs, 'cuda')
    grids = ort.OrtValue.ortvalue_from_numpy(grids, 'cuda')
    masks = ort.OrtValue.ortvalue_from_numpy(masks, 'cuda')

    bind = ort.IOBinding(session)
    bind.bind_ortvalue_input('imgs', imgs)
    bind.bind_ortvalue_input('grids', grids)
    bind.bind_ortvalue_input('masks', masks)
    bind.bind_output('inv_dist', 'cuda')

    # Single prediction save image

    # output = session.run(['inv_dist'], {'imgs': imgs, 'grids': grids, 'masks': masks})
    session.run_with_iobinding(bind)
    output = bind.copy_outputs_to_cpu()

    preds = output[0]

    pred = preds[0, 0]

    # Normalize. TODO: fix range
    pred_min, pred_max = pred.min(), pred.max()
    pred = (pred.astype(np.float32) - pred_min) / ( pred_max - pred_min )
    pred = np.clip(pred, 0, 1) * 255
    pred = cv2.applyColorMap(pred.astype(np.uint8),cv2.COLORMAP_JET)

    cv2.imwrite('test_output/backport_out.png', pred)

    # Timing

    elapsed_times = []
    for i in range(101):
            start = time.time()
            # output = session.run(['inv_dist'], {'imgs': imgs, 'grids': grids, 'masks': masks})
            session.run_with_iobinding(bind)
            elapsed = time.time() - start
            elapsed_times.append(elapsed)

    avg_elapsed = sum(elapsed_times[1:])/(len(elapsed_times) - 1)

    print('Elapsed time:', avg_elapsed)

if __name__ == '__main__':
      main()