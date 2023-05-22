#!/bin/sh
module load mpi/openmpi-4.1.3
gcc -o main main.c -lm -fopenmp
srun --reservation=fri-vr --partition=gpu --cpus-per-task=1 main