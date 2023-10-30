#!/bin/bash

# CR stands for Code Release
EV_NAME="CR_EV004"

# Always use a single GPU for offline validation.
GPU_ID=0

# Docker image.
DOCKER_IMAGE="theairlab/dsta_ngc_x86:22.08_11_lightning_for_mvs"

docker run \
    -it \
    --rm \
    --volume="/path/to/dataset/:/dataset/" \
    --volume="/path/to/mvs_gi/:/script/" \
    --volume="/path/to/root/of/working/directories/:/working_root/" \
    --env="NVIDIA_VISIBLE_DEVICES=${GPU_ID}" \
    --env="CUDA_VISIBLE_DEVICES=${GPU_ID}" \
    --gpus="device=${GPU_ID}" \
    --network host \
    --ipc host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --privileged \
    --group-add audio \
    --group-add video \
    --name dsta_${EV_NAME} \
    ${DOCKER_IMAGE} \
        /bin/bash /working_root/${EV_NAME}/eval.sh

echo "Out of the container. "
