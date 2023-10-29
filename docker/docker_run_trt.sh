#! /bin/bash

docker run \
    -it \
    --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=6710886 \
    --env-file api_key.env \
    -v `pwd`:/workspace \
    -v /home/jehon/Development/DSTA_MVS_Dataset_V2:/dataset/DSTA_MVS_Dataset_V2 \
    -w /workspace \
    jehontan/mvs-lightning:trt \
    /bin/bash