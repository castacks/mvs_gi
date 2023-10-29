#!/bin/bash

#echo commands to stdout
set -x

TASK_ID=$1
echo "Job Started!"
echo $TASK_ID
echo "-----------------"

cd /workspace

# This is Yaoyu's key.
export WANDB_API_KEY="68c672db64b5d0269500f1e75b9b93de457d0512"

python3 main_sweep.py \
	-bc config_sweepbase_4gpu.yaml \
	-c experiment_configs/sweep_hp_config${TASK_ID}.yaml \
	-pm ${TASK_ID}

# END