#!/bin/bash

polygraphy convert model.onnx -o model.engine

polygraphy convert --ckpt "wandb_logs/dsta-mvs-refactor/uwz3pc1j/checkpoints/epoch=99-step=100.ckpt" -o model.engine