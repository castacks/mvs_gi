import sys
import os

import torch
from torch import nn

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dsta_mvs.test.utils import *

import deform_conv2d_onnx_exporter
deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()

class FeatExtProxy(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, imgs):
        og_shp = imgs.shape[:2]
        imgs   = imgs.view( ( imgs.shape[0]*imgs.shape[1], *imgs.shape[2:] ) )
        feats = self.feature_extractor(imgs)
        feats = feats.view( ( *og_shp, *feats.shape[1:] ) )

        return feats


def main():
    model = load_model('wandb_logs/dsta-mvs-refactor/fp32-benchmark/checkpoints/epoch=99-step=100.ckpt')

    model.eval().cuda()

    dataloader = make_dataloader('/dataset/DSTA_MVS_Dataset_V2')
    sample = next(iter(dataloader))

    imgs        = sample['imgs'].cuda()
    grids       = sample['grids'].cuda()
    masks       = sample['masks'].cuda()

    with torch.no_grad():
        print('Exporting feature_extractor...')

        feature_extractor_in  = (imgs, )
        feature_extractor = FeatExtProxy(model.feature_extractor)
        torch.onnx.export(
            feature_extractor,
            feature_extractor_in,
            'artifacts/components/feature_extractor.onnx',
            input_names = ['imgs'],
            output_names = ['feats'],
            opset_version = 16
        )

        feats = feature_extractor(*feature_extractor_in)

        print('Exporting cv_builder...')
        cv_builder_in = (feats, grids, masks)
        cv_builder = model.cv_builder
        torch.onnx.export(
            cv_builder,
            cv_builder_in,
            'artifacts/components/cv_builder.onnx',
            input_names = ['feats', 'grids', 'masks'],
            output_names = ['vol'],
            opset_version = 16
        )

        print('Exporting cv_regulator...')
        cv_regulator_in = (cv_builder(*cv_builder_in), )
        cv_regulator = model.cv_regulator
        torch.onnx.export(
            cv_regulator,
            cv_regulator_in,
            'artifacts/components/cv_regulator.onnx',
            input_names = ['vol'],
            output_names = ['costs'],
            opset_version = 16
        )

        print('Exporting dist_regressor...')
        dist_regressor_in = (cv_regulator(*cv_regulator_in), )
        dist_regressor = model.dist_regressor
        torch.onnx.export(
            dist_regressor,
            dist_regressor_in,
            'artifacts/components/dist_regressor.onnx',
            input_names = ['costs'],
            output_names = ['inv_dist'],
            opset_version = 16
        )

    print('Done.')

if __name__ == '__main__':
    main()