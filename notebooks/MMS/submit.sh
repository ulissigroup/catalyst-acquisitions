#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:2
#SBATCH --account=m1759
#SBATCH --qos=special
#SBATCH --job-name=hallucinate_mms
#SBATCH --image=ulissigroup/catalyst-acquisitions:dev
#SBATCH --mail-user=ktran@andrew.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH --time=08:00:00

shifter python mms.py
