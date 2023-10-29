#!/bin/bash

cd /workspace

export WANDB_API_KEY=your_wandb_api_key

python3 main.py -c config_local.yaml fit

# END