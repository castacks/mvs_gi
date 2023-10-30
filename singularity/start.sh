#!/bin/bash

singularity instance start \
    --env-file ${PROJECT}/WD/dsta/dsta_mvs_lightning_code/api_key_local.env \
    --env WANDB_API_KEY= \
    --nv \
    -B ${PROJECT}/../shared/dsta/debug_dataset_20230630/:/dataset/ \
    -B ${PROJECT}/WD/dsta/dsta_mvs_lightning_code/:/workspace/ \
    -H ${PROJECT}/WD/dsta/temp_home \
    ${PROJECT}/singularity/yaoyuh_ngc_x86_dsta_22.08_11_lightning_for_mvs.sif \
    dsta_mvs

