import os
import sys

# The path of the current Python script.
_CURRENT_PATH       = os.path.dirname(os.path.realpath(__file__))
_TOP_PATH           = os.path.join(_CURRENT_PATH, '..')

if _TOP_PATH not in sys.path:
    sys.path.insert( 0, _TOP_PATH)
    for i, p in enumerate(sys.path):
        print(f'{i}: {p}')

from abc import ( ABC, abstractmethod )

import torch
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# Local packages.
from dsta_mvs.visualization import tensor_CHW_2_cv2_img
from dsta_mvs.test.utils import make_dataloader

from .proxy import ProxyBase

class DatasetProxy(ProxyBase):
    def __init__(self, 
                 argv, 
                 preprocessed_config = False,
                 debug=False):
        super().__init__(argv=argv, preprocessed_config=preprocessed_config, debug=debug)
    
        self.df = self.raw_cfg['data']['init_args']['bf']
        self.dataset = self.get_dummy_dataloader(self.raw_cfg['data']).dataset

    def get_dummy_dataloader(self, data_config):
        data_config = data_config['init_args']
        main_data_dir_dict = data_config['data_dirs']['main']

        return make_dataloader(
            main_data_dir_dict['path'],
            data_config['bf'],
            data_config['dist_list'],
            csv_rig_rgb_suffix=main_data_dir_dict['csv_rig_rgb_suffix'],
            csv_rig_dist_suffix=main_data_dir_dict['csv_rig_dist_suffix'],
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        dataset_output_dict = self.dataset[index]
        DatasetProxy.add_numpy_imgs(dataset_output_dict)
        return dataset_output_dict

    @staticmethod
    def add_numpy_imgs(dataset_output_dict):
        imgs = dataset_output_dict['imgs'] # Tensor, [K, C, H, W], where K is the number of images.
        imgs_np = [ tensor_CHW_2_cv2_img(img) for img in imgs ]
        dataset_output_dict['imgs_np'] = imgs_np
