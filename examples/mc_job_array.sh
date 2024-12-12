#!/bin/bash

#SBATCH -J mc_pi
#SBATCH -c 1
#SBATCH --mem-per-cpu=40G
#SBATCH --partition admintest
#SBATCH --array=0-10

module purge
module load miniconda
conda activate jupyter
python ./mc_job_array.py $SLURM_ARRAY_TASK_ID


