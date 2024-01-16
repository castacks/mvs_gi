
# System packages.
import glob
import os
import re
import sys

# PyTorch.
import torch

# TODO: Do we need this?
# Enable use of Tensor Cores.
torch.set_float32_matmul_precision('high')

# PyTorch Lightning.
from lightning.pytorch.cli import LightningCLI

from dsta_mvs.lightning_callback import SaveConfigWandB

from cli_processor import (
    AdditionalArg,
    preprocess_args,
    parse_args_round_1,
    parse_args_from_config,
    set_master_port
)

def extract_epoch_num(fn: str):
    s = re.search(r'step=(\d+)\.ckpt$', fn)
    return ( int( s[1] ) if s else -1, fn )

def main():
    # ========== Parse arguments. ==========
    # python3 main_cli.py fit --config_base xxx.yaml --config_sweep xxx.yaml --config_name xxx.yaml
    argv = preprocess_args(sys.argv[2:])
    args = parse_args_round_1(argv, [
        AdditionalArg('--checkpoint_run_id', 
                      type=str, 
                      default='__NONE__', 
                      help='The wandb run id of the checkpoint file to resume from. '),
        AdditionalArg('--resume_epoch', 
                      type=str, 
                      default='__NONE__', 
                      help='The string representation of the number of epoch to resume from. '),
        ]
    )
    _, raw_cfg = parse_args_from_config(args.config, instantiate_classes=False)
    
    # ========== Set port number. ==========
    set_master_port(int(args.port_modifier))
    
    if args.checkpoint_run_id != '__NONE__':
        os.environ['WANDB_RESUME'] = 'allow'
        os.environ['WANDB_RUN_ID'] = args.checkpoint_run_id

        chkp_dir_path = os.path.join( 'wandb_logs/dsta-mvs-sweep',
                                      args.checkpoint_run_id,
                                      'checkpoints' )

        if args.resume_epoch == '__NONE__':
            chkpt_names = os.listdir(chkp_dir_path)
            chkp_path = os.path.join(chkp_dir_path, max(chkpt_names, key=extract_epoch_num))
        else:
            chkp_pattern = os.path.join(chkp_dir_path, f'epoch={args.resume_epoch}_step=*.ckpt')
            chkp_paths = glob.glob(chkp_pattern)
            assert(len(chkp_paths)==1), \
                f'More than one checkpoint files found: \n'\
                f'{chkp_paths}'
            chkp_path = chkp_paths[0]
        
        print(f'')
        print(f'========== RESUMING FROM CHECKPOINT ==========')
        print(f'Loading Training from {chkp_path}. ')
        print(f'Loading from Epoch {args.resume_epoch}')
        print(f'==============================================')
        print(f'')
        raw_cfg['ckpt_path'] = chkp_path 

    cli_args = { sys.argv[1]: raw_cfg }

    cli = LightningCLI(
        args=cli_args,
        save_config_callback=SaveConfigWandB,
        save_config_kwargs={
            "config_filename": 'config_backup.yaml',
            "overwrite": True
        }
    )

if __name__ == '__main__':
    sys.exit( main() )
    