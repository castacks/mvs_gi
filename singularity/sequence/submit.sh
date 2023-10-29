#!/bin/bash

JOBS=$1
SBATCH_ID=$2 # Initial value.
RESUME_ID=$3 
ARRAY_IDS=$4

regex="job ([[:digit:]]+)"

for ((i=2; i<=${JOBS}; i++)); do
	RES=`sbatch --depend=afterany:${SBATCH_ID} --export=RESUME_ID=${RESUME_ID} --array=${ARRAY_IDS} --job-name=sec01_j${i} train_shared.sbatch`

	if [[ ${RES} =~ ${regex} ]]; then
		SBATCH_ID=${BASH_REMATCH[1]}
		echo "SBATCH_ID=${SBATCH_ID} submitted. "
	else
		echo "Error: cannot find job id in RES={RES}"
		break
	fi

done
