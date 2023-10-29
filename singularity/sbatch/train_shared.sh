#!/bin/bash

#echo commands to stdout
set -x

TASK_ID=$1
echo "Job Started!"
echo $TASK_ID
echo "-----------------"

cd /workspace

export WANDB_API_KEY="6e03853ffa33eaa006ec3b26234fdeed1e2e5692"

python3 main_sweep.py -bc config_sweepbase_4gpu.yaml -c experiment_configs/sweep_hp_config${TASK_ID}.yaml -pm ${TASK_ID}
#python3 main_sweep.py -bc config_ckpt_reloading.yaml -c sweep_hp_blank.yaml
#python3 main_sweep.py -bc /workspace/wandb_logs/wandb/run-20230726_143208-5y8wkh9w/files/config.yaml -c sweep_hp_blank.yaml


# END