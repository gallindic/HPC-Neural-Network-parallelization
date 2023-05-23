#!/bin/bash

#SBATCH --job-name=mlp_openmp
#SBATCH --output=mlp_openmp.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=32
#SBATCH --time=5:00
#SBATCH --mem-per-cpu=2000
#SBATCH --constraint=AMD
#SBATCH --reservation=fri

srun ./mlp 1


