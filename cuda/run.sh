#!/bin/sh
module load CUDA
nvcc -o mlp2 mlp2.cu
srun --reservation=fri-vr --partition=gpu --gpus=1 mlp2