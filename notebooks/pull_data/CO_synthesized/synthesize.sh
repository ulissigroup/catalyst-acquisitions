#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --constraint=knl
#SBATCH --exclusive
#SBATCH --qos=premium
#SBATCH --job-name=synthesize
#SBATCH --mail-user=ktran@andrew.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00


source ~/miniconda3/bin/activate
conda activate gaspy

python synthesize.py
