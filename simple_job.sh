#!/bin/bash
#SBATCH --account=rrg-shahabkb
#SBATCH --ntasks=4               # number of MPI processes
#SBATCH --mem-per-cpu=1024M      # memory; default unit is megabytes
#SBATCH --time=0-00:05           # time (DD-HH:MM)
#SBATCH --mail-user=murielle.mardenli@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=output.log
#SBATCH --error=error.log

python generate_base.py