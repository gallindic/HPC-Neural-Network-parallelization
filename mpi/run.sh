#!/bin/sh
#module load OpenMPI/4.1.1-GCC-11.2.0
#srun --reservation=fri-vr --partition=gpu mpicc -lm -o main main.c
#srun --reservation=fri-vr --partition=gpu --mpi=pmix --nodes=1 --ntasks=2 ./main
module load mpi/openmpi-4.1.3
mpicc -fopenmp main.c -O2 -o main -lm -g -O0
srun --time=00:10 --reservation=fri-vr --partition=gpu --mpi=pmix --nodes=1 --ntasks=2 main