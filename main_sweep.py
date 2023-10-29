

import yaml
import copy
import argparse
import os
import re
import glob

from lightning import Trainer, LightningModule
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

from dsta_mvs.model.mvs_model.spherical_sweep_stereo import *
from dsta_mvs.model.feature_extractor import *
from dsta_mvs.model.cost_volume_builder import *
from dsta_mvs.model.cost_volume_regulator import *
from dsta_mvs.model.distance_regressor import *

from dsta_mvs.support.loss_function import *
from dsta_mvs.support.augmentation import *
from dsta_mvs.support.datamodule import *

# Enable use of Tensor Cores
torch.set_float32_matmul_precision('high')

class SaveConfigWandB(SaveConfigCallback):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)

        config_dict = self.config.as_dict()
        trainer.logger.log_hyperparams(config_dict)


def replace_at_nested_keys(base_dict, key_list, value):

    _k = key_list[0]
    if len(key_list) == 1:
        base_dict[_k] = value
        return base_dict
    else:
        base_dict[_k] = replace_at_nested_keys(base_dict[_k], key_list[1:], value)
        return base_dict


def crawl_for_keyword(nested_dict, value, prepath=()):
    for k, v in nested_dict.items():
        path = tuple(prepath) + (k,)
        if v == value: # found value
            yield path
        elif hasattr(v, 'items'): # v is a dict
            yield from crawl_for_keyword(v, value, path) 


def yaml_wrapper_rpl_at_nested_keys(v, base_dict, key_list):
    if isinstance(v, str) and v.endswith(".yaml"):
        with open(v, "r") as stream:
            param_dict = yaml.safe_load(stream)
            return replace_at_nested_keys(base_dict, key_list, param_dict)
    else:
        return replace_at_nested_keys(base_dict, key_list, v)


def build_sweep_config_dict(wand_sweep_cfg, base_config, rpl_prefix="sweep@"):
    new_config = copy.deepcopy(base_config)

    #Load all sub-yamls into the base_config
    rpl_pairs = list()
    for _k, v in wand_sweep_cfg.items():
        if _k.startswith(rpl_prefix):
            rpl_pairs.append((_k, v))
        else:
            key_list = _k.split(".")
            new_config = yaml_wrapper_rpl_at_nested_keys(v, new_config, key_list)
    
    #After all sub-yamls have been loaded, replace rpl_prefix keywords with values.
    for _k, v in rpl_pairs: 
        paths = [*crawl_for_keyword(new_config, _k)]
        for p in paths:
            new_config = yaml_wrapper_rpl_at_nested_keys(v, new_config, p)

    return new_config

def extract_epoch_num(fn):
    s = re.search(r'step=(\d+)\.ckpt$', fn)
    return ( int( s[1] ) if s else -1, fn )

def main(args):
    # 65535 is the max port number supported by the operating system
    # But we want to use a 5 digit port number that starts with 20.
    port_modifier = args.port_modifier % 1000
    os.environ['MASTER_PORT'] = f'20{port_modifier:03d}'

    with open(args.base_config, "r") as base_cfg_fn:
        base_cfg = yaml.safe_load(base_cfg_fn)

    with open(args.config, "r") as wandb_cfg_fn:
        wand_sweep_cfg = yaml.safe_load(wandb_cfg_fn)

    if wand_sweep_cfg is not None:
        sweep_config_dict = build_sweep_config_dict(wand_sweep_cfg, base_cfg)
    else:
        print("No keywords found in -c path, using base config.")
        sweep_config_dict = base_cfg


    if args.checkpoint_run_id != "__NONE__":
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_RUN_ID"] = args.checkpoint_run_id

        chkp_dir_path = os.path.join("wandb_logs/dsta-mvs-sweep",args.checkpoint_run_id,"checkpoints")

        if args.resume_epoch == "__NONE__":
            chkpt_names = os.listdir(chkp_dir_path)
            chkp_path = os.path.join(chkp_dir_path, max(chkpt_names, key=extract_epoch_num))
        else:
            chkp_pattern = os.path.join(chkp_dir_path, f"epoch={args.resume_epoch}_step=*.ckpt")
            chkp_paths = glob.glob(chkp_pattern)
            assert(len(chkp_paths)==1), print(chkp_paths)
            chkp_path = chkp_paths[0]
            

        print(f"")
        print(f"========== RESUMING FROM CHECKPOINT ==========")
        print(f"Loading Training from {chkp_path}. ")
        print(f"Loading from Epoch {args.resume_epoch}")
        print(f"==============================================")
        print(f"")
        sweep_config_dict["fit"]["ckpt_path"] = chkp_path 

    cli = LightningCLI(
        args = sweep_config_dict,
        save_config_callback=SaveConfigWandB,
        save_config_kwargs={
            "config_filename": 'config_backup.yaml',
            "overwrite": True
        }
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required = True)
    parser.add_argument("-bc", "--base_config", type=str, required = True)
    parser.add_argument("-pm", "--port_modifier", type=int, required = True)
    parser.add_argument("-ckpt", "--checkpoint_run_id", type=str, default="__NONE__")
    parser.add_argument("-rep", "--resume_epoch", type=str, default="__NONE__")

    args = parser.parse_args()

    main(args)