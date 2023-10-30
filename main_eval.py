
# System packages.
import copy
from jsonargparse import ArgumentParser
import os
import sys
from typing import List

# PyTorch.
import torch

# TODO: Do we need this?
# Enable use of Tensor Cores.
torch.set_float32_matmul_precision('high')

# PyTorch Lightning.
from lightning import Trainer, LightningModule
from lightning.pytorch.cli import SaveConfigCallback

# Project packages.
from dsta_mvs.model.mvs_model.spherical_sweep_stereo import *

from dsta_mvs.model.feature_extractor import *
from dsta_mvs.model.cost_volume_builder import *
from dsta_mvs.model.cost_volume_regulator import *
from dsta_mvs.model.distance_regressor import *

from dsta_mvs.support.loss_function import *
from dsta_mvs.support.augmentation import *
from dsta_mvs.support.datamodule import *

from config_helper import ( 
        construct_config_on_filesystem,
        remove_from_args_make_copy )

# File level constants.
DEFAULT_CHECKPOINT_FN = '__NONE__'

class SaveConfigWandB(SaveConfigCallback):
    def setup(self, trainer: Trainer, 
              pl_module: LightningModule, # pl for "PyTorch Lightning".
              stage: str) -> None:
        super().setup(trainer, pl_module, stage)

        config_dict = self.config.as_dict()
        trainer.logger.log_hyperparams(config_dict)

def preprocess_args(argv: List[str]) -> str:
    '''Preprocess command line arguments. argv, the list of arguments goes through the following
    processing steps:
    1. Values of the arguments --config_base, --config_sweep, and --config_name are used to
    construct a config file which is saved to the filesystem. 
    2. The above arguments are removed from argv and saved to a new argument list.
    3. The filename of the above generated config file is appended to the new argument list as
    the --config argument.
    4. The new argument list is returned as a list of strings.

    The current implementation of this function assumes that there is a top-level "fit" key for 
    the YAML config file specified by the --config_base argument. This function only use the 
    contents under the "fit" key. An example of the YAML content of the --config_base argument 
    could be found at TODO: add link.

    Arguments:
    argv: The list of arguments to be processed.

    Returns:
    A list of processed arguments as list of strings.
    '''

    # Local constants.
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
    
    # Append a --config argument to the list of arguments.
    new_args.append( '--config' )
    new_args.append( config_name )
    
    return new_args

def parse_args_round_1(argv: List[str]):
    '''Use jsonargparse to parse the arguments that might have been preprocessed by 
    preprocess_args().

    This function returns the results from ArgumentParser.parse_args() defined by the 
    jsonargparse package.
    '''
    global DEFAULT_CHECKPOINT_FN

    parser = ArgumentParser()
    parser.add_argument('--port_modifier', type=int, default=0, 
                        help='An integer for settingg an unique port number such that '
                             'multiple training/validation processes can run on the '
                             'same machine.')
    parser.add_argument('--checkpoint_fn', type=str, default=DEFAULT_CHECKPOINT_FN,
                        help='The filename of the checkpoint. Do not specify this if the use '
                             'does not want to load a checkpoint.')
    parser.add_argument('--config', type=str, required=True,
                        help='The filename of the config file. Note that this may be generated '
                             'on the fly.')
    return parser.parse_args(args=argv)

def parse_args_from_config(fn: str):
    '''This function applies the magic of jsonargparse to instantiate class objects from a config
    file. Currently only the following keys are parsed as class objects: trainer, data, model, 
    and optimizer. 

    This function returns the instantiated class objects as well as the raw arguments as a 
    dictionary. 

    Note that jsonargparse may fail and present the user with an error message like the 
    following: TODO: add error message sample.

    This is typically due to a wrong class name or a wrong argument used for instantiating a 
    class object. It could be very difficult to debug. All of our example config have been 
    tested. So a more safe way to create a new config file is first copy an existing example and
    modify it fo the user's need.
    '''
    parser = ArgumentParser()
    
    parser.add_class_arguments(Trainer, 'trainer')
    parser.add_argument('--data', type=Any)
    parser.add_argument('--model', type=Any)
    parser.add_argument('--optimizer', type=Any)
    
    raw_cfg = parser.parse_path(fn)
    cfg = parser.instantiate_classes(raw_cfg)

    return cfg, raw_cfg

def set_master_port(port_modifier: int):
    '''65535 is the max port number supported by our Linux system. This function generates a 
    5 digit port number that starts with 20. port_modifier % 1000 is first computed internally. 
    '''
    port_modifier = port_modifier % 1000
    os.environ['MASTER_PORT'] = f'20{port_modifier:03d}'

def prepare_model(checkpoint_fn, cfg_object, cfg_raw):
    '''Create a PyTorch Lightning model by reading from a checkpoint. Two additional 
    modifications are applied to the model:
    1. The distance candidates are updated according to the configuration file. This could be 
    different from the candidates used during training.
    2. Various configurations related to dataset, validation, and visualization are overrided 
    according to the configuration file. These configurations are first recovered from the 
    checkpoint. The overrided is necessary to use the specified values in the current 
    configuration files.

    The function returns the PyTorch Lightning model.
    '''
    global DEFAULT_CHECKPOINT_FN

    if checkpoint_fn != DEFAULT_CHECKPOINT_FN:
        # Load the checkpoint.
        ckpt = torch.load(checkpoint_fn)

        # Update the dist candidates.
        ckpt['hyper_parameters']['dist_regressor'].update_dist_cands(cfg_object.data.dist_list)
        
        # Make a new model by reading the checkpoint again and the updated dist regressor.
        # TODO: Currently, this is not working. The model.dist_regressor is then overrided 
        # explicitly.
        MODULE_CLASS = cfg_object.model.__class__
        model = MODULE_CLASS.load_from_checkpoint(
            checkpoint_fn,
            # dist_regressor=ckpt['hyper_parameters']['dist_regressor'] # This does not work, may need to use self.hparams
        )
        model.dist_regressor = ckpt['hyper_parameters']['dist_regressor']
    else:
        model = cfg_object.model
    
    # Override model configurations that are changed during validation w.r.t. training.
    model.re_configure( val_loader_names=cfg_raw['model']['init_args']['val_loader_names'],
                        visualization_range=cfg_raw['model']['init_args']['visualization_range'],
                        val_offline_flag=cfg_raw['model']['init_args']['val_offline_flag'],
                        val_custom_step=cfg_raw['model']['init_args']['val_custom_step'] )
    
    return model

def prepare_datamodule(cfg_object):
    '''This function generate a PyTorch Lightning Datamodule from the configuration file. The 
    datamodule is setup for validation. Camera models are gathered and returned as a list. The 
    index of this list corresponds to the order of the datasets.

    TODO: mention is the main document that CUDA must be available. 
    '''
    datamodule = cfg_object.data

    # Force datamodule to go through the setup procedure.
    # Such that member variables like "val_datasets" are initialized.
    datamodule.setup(stage='validate')

    # Get the camera models for creating the warped input images.
    dataset_indexed_cam_models = []
    for val_dataset_dict in datamodule.val_datasets:
        dataset = val_dataset_dict['dataset']
        dataset_indexed_cam_models.append( copy.deepcopy( dataset.map_camera_model ) )

    for cam_models in dataset_indexed_cam_models:
        for _, cam_model in cam_models.items():
            cam_model.device = 'cuda' # Assuming single GPU.

    return datamodule, dataset_indexed_cam_models

def main():
    # ========== Parse arguments. ==========
    argv = preprocess_args(sys.argv[1:])
    args = parse_args_round_1(argv)
    cfg, raw_cfg = parse_args_from_config(args.config)
    
    # ========== Set port number. ==========
    set_master_port(int(args.port_modifier))
    
    # ========== PyTorch Lightning Model. ==========
    model = prepare_model(args.checkpoint_fn, cfg, raw_cfg)

    # ========== Datamodule. ==========
    datamodule, dataset_indexed_cam_models = prepare_datamodule(cfg)
    model.assign_dataset_indexed_cam_models( dataset_indexed_cam_models )
    
    # ========== Trainer. ==========
    trainer = cfg.trainer

    # ========== Validation. ==========
    trainer.validate(model=model, datamodule=datamodule)

if __name__ == '__main__':
    main()
    