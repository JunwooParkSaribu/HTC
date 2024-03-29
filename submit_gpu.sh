#!/bin/bash

#SBATCH --job-name=h2b_traj
#SBATCH --account=histoneclf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:7g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time 48:00:00

module load cudatoolkit/11.6.0
module load tensorflow/2.6.2

python3 Training_main.py

echo "Job submit done"
