
import os
import sys

# The path of the current Python script.
_CURRENT_PATH       = os.path.dirname(os.path.realpath(__file__))
_TOP_PATH           = os.path.join(_CURRENT_PATH, '..')

if _TOP_PATH not in sys.path:
    sys.path.insert( 0, _TOP_PATH)
    for i, p in enumerate(sys.path):
        print(f'{i}: {p}')

import copy
import numpy as np

import torch
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# Local packages.
from dsta_mvs.test.utils import make_dataloader

from .inference_class import InferenceProxy

class InferencePytorch(InferenceProxy):
    def __init__(self, 
                 argv, 
                 preprocessed_config=False,
                 sample_input=False,
                 debug=False):
        super().__init__(argv=argv, preprocessed_config=preprocessed_config, debug=debug)
        
        # Get the model. Override parent's model member.
        self.model = self.get_model( 
            checkpoint_fn=InferenceProxy.find_checkpoint_from_argv(argv) )
        
        self.sample_input = sample_input
        self.map_samplers = None
        if self.sample_input:
            self.prepare_samplers()

    def prepare_samplers(self):
        # Copy the samplers from the dataset.
        self.map_samplers = copy.deepcopy(self.dataloader.dataset.map_sampler)
        # Push the samplers to GPU.
        for _, sampler in self.map_samplers.items():
            sampler.device = 'cuda'

    # Override.
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

        if self.sample_input:
            sampled_imgs = []

            # NOTE: Assume batch size is always 1.
            assert t_imgs.shape[0] == 1, f'Batch size must be 1. Got {t_imgs.shape[0]} instead. '
            for i, img in enumerate(t_imgs[0, ...]):
                key = f'cam{i}'
                sampler = self.map_samplers[key]
                # NOTE: The second return value is the mask.
                sampled, _ = sampler(img)
                # NOTE: sampler(img) returns the first tensor with the batch dimension.
                sampled_imgs.append( sampled )

            t_imgs = torch.cat(sampled_imgs, dim=0).unsqueeze(0)

        return t_imgs
        
    def get_model(self, **kwargs):
        '''
        kwargs must have "checkpoint_fn" key
        '''
        checkpoint_fn = kwargs['checkpoint_fn']

        assert(os.path.isfile(checkpoint_fn)), f'Checkpoint file {checkpoint_fn} does not exist. '

        # Load the checkpoint.
        ckpt = torch.load(checkpoint_fn)

        # Update the dist candidates.
        ckpt['hyper_parameters']['dist_regressor'].update_dist_cands(self.cfg.data.dist_list)
        
        # Make a new model by reading the checkpoint again and the updated dist regressor.
        # TODO: Currently, this is not working. The model.dist_regressor is then overrided 
        # explicitly.
        MODULE_CLASS = self.cfg.model.__class__
        model = MODULE_CLASS.load_from_checkpoint(
            checkpoint_fn,
            # dist_regressor=ckpt['hyper_parameters']['dist_regressor'] # This does not work, may need to use self.hparams
        )

        model.dist_regressor = ckpt['hyper_parameters']['dist_regressor']

        # Override model configurations that are changed during validation w.r.t. training.
        model_init_args = self.raw_cfg['model']['init_args']
        model.re_configure( val_loader_names=model_init_args['val_loader_names'],
                            visualization_range=model_init_args['visualization_range'],
                            val_offline_flag=model_init_args['val_offline_flag'],
                            val_custom_step=model_init_args['val_custom_step'] )

        model.eval()
        model.cuda()

        return model
    
    def inference(self, imgs, input_dict=None):
        return self.model( 
            imgs=imgs,
            grids=self.grids,
            grid_masks=self.grid_masks,
            masks=self.masks
        )
