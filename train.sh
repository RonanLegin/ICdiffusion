#!/bin/bash
#SBATCH -p gpu -C h100
#SBATCH --gpus-per-node=4       # Request GPU "generic resources"
#SBATCH --cpus-per-task=24  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=128G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-023:30:00     # DD-HH:MM:SS

module purge
export MODULEPATH=/mnt/home/gkrawezik/modules/rocky8:$MODULEPATH
module load modules/2.1 cuda/12.0 cudnn/cuda12-8.8.0

source ~/envs/score_pytorch_h100/bin/activate



python train.py
