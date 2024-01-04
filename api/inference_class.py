
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

import numpy as np

import torch
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# Local packages.
from dsta_mvs.test.utils import make_dataloader

from .proxy import ProxyBase

class InferenceProxy(ProxyBase, ABC):
    def __init__(self, 
                 argv, 
                 preprocessed_config = False,
                 debug=False):
        super().__init__(argv=argv, preprocessed_config=preprocessed_config, debug=debug)
    
        self.bf = self.raw_cfg['data']['init_args']['bf']

        # Get the dummy dataset.
        self.dummy_dataloader = self.get_dummy_dataloader(self.raw_cfg['data'])
        
        # Move the sampling grids to GPU. And add the batch dimension.
        batch = next(iter(self.dummy_dataloader))

        self.dummy_imgs = batch['imgs'].cuda()
        self.grids      = batch['grids'].cuda()
        self.grid_masks = batch['grid_masks'].cuda()
        self.masks      = batch['masks'].cuda()
        
        # Get the model.
        self.model = None # Could be PyTorch model or TensorRT engine.
    
    @property
    def dataloader(self):
        return self.dummy_dataloader
    
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
    
    @abstractmethod
    def get_model(self, model_config, **kwargs):
        pass
    
    def preprocess_imgs(self, imgs):
        if isinstance(imgs, torch.Tensor):
            # NOTE: This is the debug branch.
            t_imgs = imgs.cuda()
        else:
            if isinstance(imgs, list):
                imgs = np.stack(imgs, axis=0)
            # Convert the input from NumPy to Tensor.
            t_imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda().unsqueeze(0)
            if t_imgs.dtype == torch.uint8:
                t_imgs = t_imgs.float() / 255.0

        return t_imgs

    def postprocess_imgs(self, inv_dist):
        with torch.no_grad():
            inv_dist = inv_dist / self.bf
        return inv_dist.detach().squeeze(0).squeeze(0).to('cpu').numpy()

    @abstractmethod
    def inference(self, imgs, input_dict=None):
        pass

    def __call__(self, input_dict):
        '''
        input_dict must have 'imgs' key.
        '''
        t_imgs = self.preprocess_imgs(input_dict['imgs'])
        preds = self.inference(t_imgs, input_dict=input_dict)
        preds = self.postprocess_imgs(preds)
        return preds
