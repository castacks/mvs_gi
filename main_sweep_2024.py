
# System packages.
import copy
import glob
from jsonargparse import ArgumentParser
import os
import re
import sys
from typing import Any, List

# PyTorch.
import torch

# TODO: Do we need this?
# Enable use of Tensor Cores
torch.set_float32_matmul_precision('high')

# PyTorch Lightning.
from lightning import Trainer, LightningModule
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

from config_helper import ( 
        construct_config_on_filesystem,
        remove_from_args_make_copy )

class SaveConfigWandB(SaveConfigCallback):
    def setup(self, trainer: Trainer, 
              pl_module: LightningModule, 
              stage: str) -> None:
        super().setup(trainer, pl_module, stage)

        config_dict = self.config.as_dict()
        trainer.logger.log_hyperparams(config_dict)

def extract_epoch_num(fn):
    s = re.search(r'step=(\d+)\.ckpt$', fn)
    return ( int( s[1] ) if s else -1, fn )

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

    parser = ArgumentParser()
    parser.add_argument('--port_modifier', type=int, default=0, 
                        help='An integer for settingg an unique port number such that '
                             'multiple training/validation processes can run on the '
                             'same machine.')
    parser.add_argument('--checkpoint_run_id', type=str, default='__NONE__',
                        help='The wandb run id of the checkpoint. Do not specify this if the '
                             'user does not want to load a checkpoint.')
    parser.add_argument('--resume_epoch', type=str, default='__NONE__',
                        help='The epoch number of the checkpoint to resume from. If the user '
                             'does not specify this, but --checkpoint_run_id is set. Then the '
                             'latest checkpoint will be used.')
    parser.add_argument('--config', type=str, required=True,
                        help='The filename of the config file. Note that this may be generated '
                             'on the fly.')
    return parser.parse_args(args=argv)

def parse_args_from_config(fn: str, instantiate_classes=True):
    '''When instanticate_classes=True, then this function applies the magic of jsonargparse to
    instantiate class objects from a config file. Currently only the following keys are parsed 
    as class objects: trainer, data, model, and optimizer. 

    This function returns the instantiated class objects as well as the raw arguments as a 
    dictionary. 

    Note that jsonargparse may fail and present the user with an error message like the 
    following: TODO: add error message sample.

    This is typically due to a wrong class name or a wrong argument used for instantiating a 
    class object. It could be very difficult to debug. All of our example config have been 
    tested. So a more safer way to create a new config file is first copy an existing example and
    modify it fo the user's need.
    '''
    parser = ArgumentParser()
    
    parser.add_class_arguments(Trainer, 'trainer')
    parser.add_argument('--data', type=Any)
    parser.add_argument('--model', type=Any)
    parser.add_argument('--optimizer', type=Any)
    
    raw_cfg = parser.parse_path(fn)

    cfg = parser.instantiate_classes(raw_cfg) \
        if instantiate_classes \
        else None

    return cfg, raw_cfg

def set_master_port(port_modifier: int):
    '''65535 is the max port number supported by our Linux system. This function generates a 
    5 digit port number that starts with 20. port_modifier % 1000 is first computed internally. 
    '''
    port_modifier = port_modifier % 1000
    os.environ['MASTER_PORT'] = f'20{port_modifier:03d}'

def augment_cfg_raw_by_checkpoint( cfg_raw, 
                                   checkpoint_run_id='__NONE__', 
                                   resume_epoch='__NONE__' ):
    '''Add "ckpt_path" key to the cfg_raw dictionary if there is a valid checkpoint.

    Side effectt: cfg_raw is modified in place.
    '''
    if checkpoint_run_id != '__NONE__':
        os.environ['WANDB_RESUME'] = 'allow'
        os.environ['WANDB_RUN_ID'] = checkpoint_run_id

        chkp_dir_path = os.path.join( 
            'wandb_logs/dsta-mvs-sweep', checkpoint_run_id, 'checkpoints' )

        if resume_epoch == '__NONE__':
            chkpt_names = os.listdir( chkp_dir_path )
            chkp_path   = os.path.join( 
                chkp_dir_path, max( chkpt_names, key=extract_epoch_num ) )
        else:
            chkp_pattern = os.path.join( 
                chkp_dir_path, f'epoch={resume_epoch}_step=*.ckpt' )
            chkp_paths   = glob.glob(chkp_pattern)
            assert( len(chkp_paths)==1 ), print(chkp_paths)
            chkp_path = chkp_paths[0]
            

        print(f'')
        print(f'========== RESUMING FROM CHECKPOINT ==========')
        print(f'Loading Training from {chkp_path}. ')
        print(f'Loading from Epoch {resume_epoch}. ')
        print(f'==============================================')
        print(f'')

        if 'fit' in cfg_raw:
            cfg_raw['fit']['ckpt_path'] = chkp_path
        else:
            cfg_raw['ckpt_path'] = chkp_path

def main():
    # ========== Parse arguments. ==========
    argv = preprocess_args(sys.argv[1:])
    args = parse_args_round_1(argv)
    _, raw_cfg = parse_args_from_config(args.config, instantiate_classes=False)
    
    # ========== Set port number. ==========
    set_master_port( int(args.port_modifier) )
    
    # ========== Augment the config if continue from a checkpoint. ==========
    augment_cfg_raw_by_checkpoint(raw_cfg, args.checkpoint_run_id, args.resume_epoch)

    lightning_args = raw_cfg if 'fit' in raw_cfg else { 'fit': raw_cfg }

    cli = LightningCLI(
        args = lightning_args,
        save_config_callback=SaveConfigWandB,
        save_config_kwargs={
            "config_filename": 'config_backup.yaml',
            "overwrite": True
        }
    )

if __name__ == '__main__':
    main()
