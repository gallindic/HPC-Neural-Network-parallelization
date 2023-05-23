gcc -fopenmp -lm -o mlp mlp.c
srun --reservation=fri --cpus-per-task=1 ./mlp 1