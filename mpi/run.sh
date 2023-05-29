#!/bin/sh
module load mpi/openmpi-4.1.3
mpicc -o main main.c -lm
srun --reservation=fri-vr --partition=gpu --mpi=pmix --nodes=1 --ntasks=2 ./main