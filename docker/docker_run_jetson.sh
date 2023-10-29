#! /bin/bash

docker run \
    -it \
    --rm \
    --runtime nvidia \
    --network host \
    --ipc=host \
    --env-file api_key.env \
    -v `pwd`:/workspace \
    -v /data/Projects/OpenResearchDrone/MultiviewStereo/DSTA_MVS_Dataset_V2:/dataset/DSTA_MVS_Dataset_V2 \
    -w /workspace \
    jehontan/mvs-jetson:jetpack-4.6.1 \
    /bin/bash