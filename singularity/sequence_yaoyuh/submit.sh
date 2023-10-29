#!/bin/bash

JOBS=$1 # To total number of jobs INCLUDING the first one, which is already running.
SBATCH_ID=$2 # SLURM job ID for the first job.
RESUME_ID=$3 # wandb run ID.
ARRAY_IDS=$4 # Config number.

regex="job ([[:digit:]]+)"

for ((i=2; i<=${JOBS}; i++)); do
	RES=`sbatch --depend=afterany:${SBATCH_ID} --export=RESUME_ID=${RESUME_ID} --array=${ARRAY_IDS} --job-name=cr_j${i} train_shared.sbatch`

	if [[ ${RES} =~ ${regex} ]]; then
		SBATCH_ID=${BASH_REMATCH[1]}
		echo "SBATCH_ID=${SBATCH_ID} submitted. "
	else
		echo "Error: cannot find SLURM job ID in RES={RES}"
		break
	fi

done
