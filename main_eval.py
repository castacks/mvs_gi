
import copy
from jsonargparse import ArgumentParser
import os
import re
import sys

import torch

from lightning import Trainer, LightningModule
from lightning.pytorch.cli import SaveConfigCallback

from dsta_mvs.model.mvs_model.spherical_sweep_stereo import *
from eval_wrappers.rtss_wrapper_yaoyuh import RealTimeSphereSweepWrapper

from dsta_mvs.model.feature_extractor import *
from dsta_mvs.model.cost_volume_builder import *
from dsta_mvs.model.cost_volume_regulator import *
from dsta_mvs.model.distance_regressor import *

from dsta_mvs.support.loss_function import *
from dsta_mvs.support.augmentation import *
from dsta_mvs.support.datamodule import *

from config_helper import ( 
        gather_args, remove_from_args_make_copy, construct_config_on_filesystem )

# Enable use of Tensor Cores
torch.set_float32_matmul_precision('high')

class SaveConfigWandB(SaveConfigCallback):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)

        config_dict = self.config.as_dict()
        trainer.logger.log_hyperparams(config_dict)

def extract_epoch_num(fn):
    s = re.findall("\d+$", fn)
    return (int(s[0]) if s else -1,fn)

def preprocess_args(argv):
    name_config_base  = '--config_base'
    name_config_sweep = '--config_sweep'
    name_config_name  = '--config_name'
    
    config_name = construct_config_on_filesystem(
        argv,
        top_level_single_key='fit',
        name_config_base=name_config_base,
        name_config_sweep=name_config_sweep,
        name_config_name=name_config_name )
    
    new_args = remove_from_args_make_copy(argv, 
            { name_config_base:  1, 
              name_config_sweep: 1,
              name_config_name:  1 } )
    
    new_args.append( '--config' )
    new_args.append( config_name )
    
    return new_args

def parse_args_round_1(argv):
    parser = ArgumentParser()
    parser.add_argument('--port_modifier', type=int, default=0)
    parser.add_argument('--checkpoint_fn', type=str, default='')
    parser.add_argument('--config', type=str) 
    # parser.add_argument('--rtss', action='store_true', default='False')
    return parser.parse_args(args=argv)

def parse_args_from_config(fn):
    parser = ArgumentParser()
    
    parser.add_class_arguments(Trainer, 'trainer')
    parser.add_argument('--data', type=Any)
    parser.add_argument('--model', type=Any)
    parser.add_argument('--optimizer', type=Any)
    
    raw_cfg = parser.parse_path(fn)
    cfg = parser.instantiate_classes(raw_cfg)
    return cfg, raw_cfg

def main():
    argv = preprocess_args(sys.argv[1:])
    
    args = parse_args_round_1(argv)
    cfg, raw_cfg = parse_args_from_config(args.config)
    
    # 65535 is the max port number supported by the operating system
    # But we want to use a 5 digit port number that starts with 20.
    port_modifier = int(args.port_modifier) % 1000
    os.environ['MASTER_PORT'] = f'20{port_modifier:03d}'
    
    if args.checkpoint_fn != '__NONE__':
        # Load the checkpoint.
        ckpt = torch.load(args.checkpoint_fn)
        ckpt['hyper_parameters']['dist_regressor'].update_dist_cands(cfg.data.dist_list)
        # update_regresser_inv_dist_idx( 
        #     ckpt['hyper_parameters']['dist_regressor'], 
        #     np.array(cfg.data.dist_list) )
        
        # ========== Model. ==========
        # Make a new model using the checkpoint and the updated dist regressor.
        MODULE_CLASS = cfg.model.__class__
        model = MODULE_CLASS.load_from_checkpoint(
            args.checkpoint_fn,
            # dist_regressor=ckpt['hyper_parameters']['dist_regressor'] # This does not work, may need to use self.hparams
        )

        model.dist_regressor = ckpt['hyper_parameters']['dist_regressor']
    else:
        model = cfg.model
    model.re_configure( val_loader_names=raw_cfg['model']['init_args']['val_loader_names'],
                            visualization_range=raw_cfg['model']['init_args']['visualization_range'],
                            val_offline_flag=raw_cfg['model']['init_args']['val_offline_flag'],
                            val_custom_step=raw_cfg['model']['init_args']['val_custom_step'])

    # ========== Datamodule. ==========
    datamodule = cfg.data
    # Force setup.
    datamodule.setup(stage='validate')
    # import ipdb; ipdb.set_trace()
    # Get the camera models for creating the warped input images.
    dataset_indexed_cam_models = []
    for val_dataset_dict in datamodule.val_datasets:
        dataset = val_dataset_dict['dataset']
        dataset_indexed_cam_models.append( copy.deepcopy( dataset.map_camera_model ) )

    for cam_models in dataset_indexed_cam_models:
        for _, cam_model in cam_models.items():
            cam_model.device = 'cuda' # Assuming single GPU.

    model.assign_dataset_indexed_cam_models( dataset_indexed_cam_models )
    
    # ========== Trainer. ==========
    trainer = cfg.trainer
    trainer.validate(model=model, datamodule=datamodule)

if __name__ == '__main__':
    main()
    