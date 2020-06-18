#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=m1759
#SBATCH --qos=special
#SBATCH --job-name=rs_cfgp
#SBATCH --image=ulissigroup/catalyst-acquisitions:dev
#SBATCH --mail-user=ktran@andrew.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH --time=08:00:00

shifter python hallucinate.py
