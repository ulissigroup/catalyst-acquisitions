#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --constraint=knl
#SBATCH --exclusive
#SBATCH --qos=premium
#SBATCH --job-name=hallucinate_tpot
#SBATCH --image=ulissigroup/catalyst-acquisitions:dev
#SBATCH --mail-user=ktran@andrew.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00

shifter python TPOT_heuristic.py
