#!/bin/bash

# echo commands to stdout
set -x

EV_ID="CR_EV004"
CONFIG_ID=103
WB_TRAIN_ID="jy2dqg6r"
CKPT_VER="v102"

echo "Evaluation Started!"
echo "CONFIG $CONFIG_ID"
echo "WB_TRAIN_ID=$WB_TRAIN_ID"
echo "-----------------"

export WANDB_API_KEY="your wandb api key"

cd /script
python3 main_eval.py \
	--port_modifier ${CONFIG_ID} \
	--config_base /script/config_sweepbase_4gpu.yaml \
	--config_sweep /script/experiment_configs/sweep_hp_config${CONFIG_ID}.yaml \
	--config_sweep /working_root/${EV_ID}/ev_sweep_dataset.yaml \
	--config_sweep /script/eval_configs/sweep_constants.yaml \
	--config_name /working_root/${EV_ID}/composed_config.yaml \
	--checkpoint_fn /working_root/${EV_ID}/dsta_sweep_config${CONFIG_ID}_WB_${WB_TRAIN_ID}_${CKPT_VER}.ckpt

# END
