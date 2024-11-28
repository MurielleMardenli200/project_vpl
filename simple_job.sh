#!/bin/bash
#SBATCH --account=rrg-shahabkb
#SBATCH --mem=60000
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH --time=0-03:00 

#SBATCH --mail-user=murielle.mardenli@gmail.com
#SBATCH --mail-type=ALL

#SBATCH --output=output.log
#SBATCH --error=error.log

python hyperparameter_sweep.py
