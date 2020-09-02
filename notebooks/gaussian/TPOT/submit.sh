#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --constraint=knl
#SBATCH --exclusive
#SBATCH --qos=premium
#SBATCH --job-name=gauspot
#SBATCH --image=ulissigroup/kubeflow_catalyst_acquisitions:latest
#SBATCH --mail-user=ktran@andrew.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00


export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

shifter python hallucinate.py
