#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --constraint=knl
#SBATCH --exclusive
#SBATCH --qos=regular
#SBATCH --job-name=tpot_hallucination
#SBATCH --image=ulissigroup/catalyst-acquisitions:dev
#SBATCH --mail-user=ktran@andrew.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00

shifter python hallucinate.py
