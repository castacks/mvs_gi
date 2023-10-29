#!/bin/bash

#SBATCH -p GPU-shared
#SBATCH --time=0-00:10:00
#SBATCH -J cr
#SBATCH --gpus=v100-32:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=5
#SBATCH -o job-%A-%a.out
#SBATCH -e job-%A-%a.err
#SBATCH --mail-user=yaoyuh@andrew.cmu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=ARRAY_TASKS

srun 'bash' train_shared.job ${SLURM_ARRAY_TASK_ID}
