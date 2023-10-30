#!/bin/bash

#echo commands to stdout
set -x

TASK_ID=$1
RESUME_ID=$2 # wandb run ID.

echo "Job Started!"
echo "CONFIG $TASK_ID"
echo "RESUME_ID=$RESUME_ID (wandb run ID)"
echo "-----------------"

cd /workspace

# This is Yaoyu's key.
export WANDB_API_KEY=""

python3 main_sweep.py \
	-bc config_sweepbase_4gpu.yaml \
	-c experiment_configs/sweep_hp_config${TASK_ID}.yaml \
	-pm ${TASK_ID} \
	-ckpt ${RESUME_ID}

# END