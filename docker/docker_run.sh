#! /bin/bash

docker run \
    -it \
    --rm \
    -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=6710886 \
    --env-file api_key.env \
    -v `pwd`:/workspace \
    -v /tmp2/DockerTmpfs_yaoyuh/Projects/OpenResearchDrone/MultiviewStereo/Processed_Data/:/dataset \
    -w /workspace \
    jehontan/mvs-lightning:latest \
    /bin/bash
