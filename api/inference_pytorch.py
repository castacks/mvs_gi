
import os
import sys

# The path of the current Python script.
_CURRENT_PATH       = os.path.dirname(os.path.realpath(__file__))
_TOP_PATH           = os.path.join(_CURRENT_PATH, '..')

if _TOP_PATH not in sys.path:
    sys.path.insert( 0, _TOP_PATH)
    for i, p in enumerate(sys.path):
        print(f'{i}: {p}')

import torch
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# Local packages.
from dsta_mvs.test.utils import make_dataloader

from .inference_class import InferenceProxy

class InferencePytorch(InferenceProxy):
    def __init__(self, 
                 argv, 
                 preprocessed_config = False,
                 debug=False):
        super().__init__(argv=argv, preprocessed_config=preprocessed_config, debug=debug)
        
        # Get the model. Override parent's model member.
        self.model = self.get_model( 
            checkpoint_fn=InferenceProxy.find_checkpoint_from_argv(argv) )
    
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
