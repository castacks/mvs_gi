import time
import os
import sys

import cv2
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dsta_mvs.test.utils import *
from dsta_mvs.model.mvs_model.torch_only import SphericalSweepStereoBase

from dsta_mvs.visualization import render_visualization

def main():
    # Load data
    dataloader = make_dataloader(
        '/dataset/DSTA_MVS_Dataset_V2', # TODO set dataset directory
        bf=96,
        dist_list=[0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100]
    )

    batch = next(iter(dataloader))

    imgs        = batch['imgs'].cuda()
    grids       = batch['grids'].cuda()
    masks       = batch['masks'].cuda()

    # Load model
    chkpt = torch.load('wandb_logs/dsta-mvs-refactor/gr6auuo8/checkpoints/epoch=99-step=100.ckpt')
    hparams = chkpt['hyper_parameters']
    model = SphericalSweepStereoBase(
        feature_extractor = hparams['feature_extractor'],
        cv_builder = hparams['cv_builder'],
        cv_regulator = hparams['cv_regulator'],
        dist_regressor = hparams['dist_regressor']
    )
    model.eval()
    model.cuda()

    with torch.no_grad():
        preds = model(imgs, grids, masks)

    pred = preds[0, 0].detach().to('cpu').numpy()

    pred = render_visualization(pred)

    cv2.imwrite('torch_out.png', pred)

    # Timing

    elapsed_times = []
    with torch.no_grad():
        for i in range(21):
            torch.cuda.synchronize()
            start = time.time()
            output = model(imgs, grids, masks)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            elapsed_times.append(elapsed)

    avg_elapsed = sum(elapsed_times[1:])/(len(elapsed_times) - 1)

    print('Elapsed time:', avg_elapsed)

if __name__ == '__main__':
    main()
