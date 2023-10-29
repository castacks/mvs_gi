#!/bin/bash

srun \
    -J Interact \
    -A $CID_CURRENT \
    -p GPU-shared \
    --gres=gpu:v100-32:1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    -t 60:00 \
    -N 1 \
    --pty zsh \
    -i

