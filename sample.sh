#!/bin/bash
#SBATCH -p gpu -C h100
#SBATCH --gpus-per-node=1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-03:00:00     # DD-HH:MM:SS
#SBATCH --array=1-4

module purge
export MODULEPATH=/mnt/home/gkrawezik/modules/rocky8:$MODULEPATH
module load openmpi python-mpi
module load modules/2.1 cuda/12.0 cudnn/cuda12-8.8.0

source ~/envs/score_pytorch_h100/bin/activate



python sample.py $SLURM_ARRAY_TASK_ID fiducial/
