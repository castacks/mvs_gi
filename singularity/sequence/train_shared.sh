#!/bin/bash

#echo commands to stdout
set -x

TASK_ID=$1
RESUME_ID=$2
RESUME_EPOCH=$3

echo "Job Started!"
echo $TASK_ID
echo "RESUME_ID passed is ${RESUME_ID}..."
echo "-----------------"

cd /workspace

export WANDB_API_KEY=""

python3 main_sweep.py -bc config_sweepbase_4gpu.yaml \
        -c experiment_configs/sweep_hp_config${TASK_ID}.yaml \
        -pm ${TASK_ID} -ckpt ${RESUME_ID} -rep ${RESUME_EPOCH}

# END