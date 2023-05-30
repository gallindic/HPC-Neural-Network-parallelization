#!/bin/sh
module load mpi/openmpi-4.1.3
mpicc -fopenmp main.c -O2 -o main -lm -g -O0
srun --time=00:10 --reservation=fri-vr --partition=gpu --mpi=pmix --nodes=1 --ntasks=2 main