#!/bin/bash
#SBATCH --account=rrg-shahabkb
#SBATCH --ntasks=4               # number of MPI processes
#SBATCH --gpus=1
#SBATCH --time=0-03:00           # time (DD-HH:MM)
#SBATCH --mail-user=murielle.mardenli@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=output.log
#SBATCH --error=error.log

python hyperparameter_sweep.py