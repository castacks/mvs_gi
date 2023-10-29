#!/bin/bash

RESUME_ID=$1
ARRAY_IDS=$2

sbatch --job-name sec01_j1 --export=RESUME_ID=${RESUME_ID} --array=${ARRAY_IDS} train_shared.sbatch
