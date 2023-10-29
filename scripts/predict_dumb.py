import time
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

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

    session = ort.InferenceSession(
            'artifacts/model_fp32.onnx',
            providers=[
                'CPUExecutionProvider'
            ])

    # Single prediction save image

    output = session.run(['inv_dist'], {'imgs': imgs, 'grids': grids, 'masks': masks})

    preds = output[0]

    pred = preds[0, 0]

    # Normalize. TODO: fix range
    pred_min, pred_max = pred.min(), pred.max()
    pred = (pred.astype(np.float32) - pred_min) / ( pred_max - pred_min )
    pred = np.clip(pred, 0, 1) * 255
    pred = cv2.applyColorMap(pred.astype(np.uint8),cv2.COLORMAP_JET)

    cv2.imwrite('dumb_out.png', pred)

    print(pred_min, pred_max)
    print('Done.')

if __name__ == '__main__':
      main()