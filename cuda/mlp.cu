#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "../datasets/dataset.h"
#include "../serial/mlp.h"

#define ETA 0.0001
#define EPOCHS 10

#define NUM_FEATURES 14 
#define HIDDEN_SIZE 1000
#define OUTPUTS 1

#define BATCH_SIZE 128
#define GRID_SIZE 32
#define BLOCK_SIZE BATCH_SIZE / GRID_SIZE

#define HIDDEN_ROWS_PER_THREAD (HIDDEN_SIZE + BATCH_SIZE - 1) / BATCH_SIZE

__device__ double cuda_tanh_activation(double x) {
    return tanh(x);
}

__device__ void calculate_hidden_layer_activation(
    const double* A, 
    const double* weights,
    const double* bias, 
    double* C
) {
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

    double sum[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        sum[i] = 0.0;
    }   

    for(int i = 0; i < NUM_FEATURES; i++) {
        for(int j = 0; j < HIDDEN_SIZE; j++) {
            sum[j] += A[(globalIdx * NUM_FEATURES) + i] * weights[i * j + j];
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        C[i] = cuda_tanh_activation(sum[i] + bias[i]);
    }  
}

__device__ void calculate_output_layer_activation(
    double* A, 
    const double* weights, 
    const double* bias, 
    double* C
) {
    double sum = 0.0;

    for(int i = 0; i < HIDDEN_SIZE; i++) {
        sum += A[i] * weights[i];
    }

    *C = cuda_tanh_activation(sum + bias[0]);  
}

__global__ void copy_batch_data(const double* X, const double* Y, double* Xb, double* Yb, int rows, int batch) {
    int rowId = threadIdx.x + blockIdx.x * blockDim.x;

    if(rowId < rows) {
        for(int i = 0; i < NUM_FEATURES; i++) {
            Xb[rowId * NUM_FEATURES + i] = X[(batch * BATCH_SIZE) + rowId * NUM_FEATURES + i];
        }

        Yb[rowId] = Y[rowId + (batch * BATCH_SIZE)];
    }
}

/**
 * Forward propagation
 * H ← tanh(XbW1 + b1)
 * Yˆ ← tanh(HW2 + b2)
 */
__global__ void forward_propagation(double* Xb, double* W1, double* W2, double* b1, double* b2, double* Y_hat_out, double* H_out) {
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

    double H_local [HIDDEN_SIZE];
    double Y_hat_local;
    
    calculate_hidden_layer_activation(Xb, W1, b1, H_local);
    calculate_output_layer_activation(H_local, W2, b2, &Y_hat_local);

    for(int i = 0; i < HIDDEN_SIZE; i++) {
        H_out[globalIdx * HIDDEN_SIZE + i] = H_local[i];
    }
 
    Y_hat_out[globalIdx] = Y_hat_local;
}

__global__ void calculate_error(double* Yb, double* Y_hat, double* E_out) {
    int rowId = threadIdx.x + blockIdx.x * blockDim.x;
    E_out[rowId] = Y_hat[rowId] - Yb[rowId];
}

__device__ void transpose1DArray(const double* input, double* output, int rows, int cols)
{
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 0; i < cols; i++) {
        output[i * rows + threadIdx.x] = input[globalIdx * cols + i];
    }
}

__device__ void calculate_output_layer_backpropagation(double* Y_hat, double* H_t, double* W2, double* b2)
{
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float val;

    if(globalIdx < OUTPUTS) {
        val = 0.0;
    }

    int rowsPerThread = HIDDEN_ROWS_PER_THREAD;
    int startRow = globalIdx * rowsPerThread;
    int endRow = min(startRow + rowsPerThread, HIDDEN_SIZE);
    
    if(startRow < HIDDEN_SIZE) {
        int idx = 0;

        for(int row = startRow; row < endRow; row++) {
            for (int col = 0; col < OUTPUTS; col++) {
                double sum = 0.0;

                for (int k = 0; k < BATCH_SIZE; k++) {
                    sum += H_t[row * BATCH_SIZE + k * blockIdx.x + k] * Y_hat[k * OUTPUTS + col];
                }

                W2[globalIdx * HIDDEN_ROWS_PER_THREAD + idx++] -= ETA * sum;
            }
        }
    }

    atomicAdd(&val, (float) Y_hat[globalIdx]);

    __syncthreads();

    if(globalIdx < OUTPUTS) {
        b2[globalIdx] = val;
    }
}

__device__ void calculate_hidden_layer_backpropagation(double* Xb_t, double E_Y_hat_hadamard, double* H, double* W2_t, double* W1, double* b1)
{
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int col = 0; col < HIDDEN_SIZE; col++) {
        double sum = 0.0;

        for (int k = 0; k < OUTPUTS; k++) {
            sum += E_Y_hat_hadamard * W2_t[col];
        }

        double H_squared_minus = 1.0 - (H[globalIdx * HIDDEN_SIZE + col] * H[globalIdx * HIDDEN_SIZE + col]);
        H[threadIdx.x * HIDDEN_SIZE + col] = sum * H_squared_minus;
    }

    if(globalIdx < NUM_FEATURES) {
        for (int col = 0; col < HIDDEN_SIZE; col++) {
            double sum = 0.0;

            for (int k = 0; k < BATCH_SIZE; k++) {
                sum += Xb_t[globalIdx * BATCH_SIZE + k] * H[k * HIDDEN_SIZE + col];
            }

            W1[globalIdx * HIDDEN_SIZE + col] -= ETA * sum;
        }
    }
}

__global__ void back_propagation(double* Xb_t, double* Y_hat, double* H, double* H_t, double* E, double* W1, double* W2, double* W2_t, double* b1, double* b2) {
    double E_Y_hat_hadamard;
    double Y_hat_squared_minus;
    
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

    Y_hat_squared_minus = 1.0 - (Y_hat[globalIdx] * Y_hat[globalIdx]);
    E_Y_hat_hadamard = E[globalIdx] * Y_hat_squared_minus;
    Y_hat[globalIdx] = E_Y_hat_hadamard;

    __syncthreads();

    calculate_output_layer_backpropagation(Y_hat, H_t, W2, b2);
    calculate_hidden_layer_backpropagation(Xb_t, E_Y_hat_hadamard, H, W2_t, W1, b1);
}

__global__ void transposeMatrices(double* Xb, double* Xb_t, double* H, double* H_t, double* W2, double* W2_t)
{
    transpose1DArray(H, H_t, BATCH_SIZE, HIDDEN_SIZE);
    transpose1DArray(W2, W2_t, HIDDEN_SIZE, OUTPUTS);
    transpose1DArray(Xb, Xb_t, BATCH_SIZE, NUM_FEATURES);
}

double* flattenDoubleArray(double** arr, int rows, int columns) {
    double* flattenedArray = (double*)malloc(sizeof(double) * rows * columns);
    
    int index = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            flattenedArray[index++] = arr[i][j];
        }
    }

    return flattenedArray;
}

int main(void) {
    DATASET dataset;
    MLP mlp;

    prepare_dataset("../datasets/adults/encoded_data_normalized.data", NUM_FEATURES, OUTPUTS, &dataset);

    init_mlp(&mlp, dataset.train_set.X, dataset.train_set.size, NUM_FEATURES, OUTPUTS, HIDDEN_SIZE);

    double *flatX = flattenDoubleArray(dataset.train_set.X, dataset.train_set.size, NUM_FEATURES);
    double *flatW1 = flattenDoubleArray(mlp.hiddenLayer.W1, NUM_FEATURES, HIDDEN_SIZE);
    double *flatW2 = flattenDoubleArray(mlp.outputLayer.W2, HIDDEN_SIZE, OUTPUTS);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double *device_X;
    double *device_Y;
    cudaMalloc((void **)&device_X, NUM_FEATURES * dataset.train_set.size * sizeof(double));
    cudaMalloc((void **)&device_Y, dataset.train_set.size * sizeof(double));
    cudaMemcpy(device_X, flatX, sizeof(double) * dataset.train_set.size * NUM_FEATURES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_Y, dataset.train_set.Y, sizeof(double) * dataset.train_set.size, cudaMemcpyHostToDevice);

    double *device_Xb;
    double *device_Yb;
    cudaMalloc((void **)&device_Xb, NUM_FEATURES * BATCH_SIZE * sizeof(double));
    cudaMalloc((void **)&device_Yb, BATCH_SIZE * sizeof(double));

    double *deviceW1;
    double *deviceW2;
    cudaMalloc((void**)&deviceW1, sizeof(double) * NUM_FEATURES * HIDDEN_SIZE);
    cudaMalloc((void**)&deviceW2, sizeof(double) * HIDDEN_SIZE * OUTPUTS);
    cudaMemcpy(deviceW1, flatW1, sizeof(double) * NUM_FEATURES * HIDDEN_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceW2, flatW2, sizeof(double) * HIDDEN_SIZE * OUTPUTS, cudaMemcpyHostToDevice);

    double *deviceB1;
    double *deviceB2;
    cudaMalloc((void**)&deviceB1, sizeof(double) * HIDDEN_SIZE);
    cudaMalloc((void**)&deviceB2, sizeof(double) * OUTPUTS);
    cudaMemcpy(deviceB1, mlp.hiddenLayer.b1, sizeof(double) * HIDDEN_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB2, mlp.outputLayer.b2, sizeof(double) * OUTPUTS, cudaMemcpyHostToDevice);

    double *device_Y_hat;
    double *device_H;
    cudaMalloc((void **)&device_Y_hat, BATCH_SIZE * sizeof(double));
    cudaMalloc((void **)&device_H, BATCH_SIZE * HIDDEN_SIZE * sizeof(double));

    double *device_H_t;
    double *device_W2_t;
    double *device_Xb_t;
    cudaMalloc((void **)&device_H_t, BATCH_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&device_W2_t, sizeof(double) * HIDDEN_SIZE * OUTPUTS);
    cudaMalloc((void **)&device_Xb_t, NUM_FEATURES * BATCH_SIZE * sizeof(double));

    double *device_E;
    cudaMalloc((void **)&device_E, BATCH_SIZE * sizeof(double));

    int batches = BATCH_SIZE >= dataset.train_set.size ? 1 : ((dataset.train_set.size + BATCH_SIZE - 1) / BATCH_SIZE);
    dim3 gridSize(GRID_SIZE);
    dim3 blockSize(BLOCK_SIZE);

    printf("Batches: %d, Block size: %d, grid size: %d\n", batches, BLOCK_SIZE, GRID_SIZE);

    cudaEventRecord(start);

    for(int epoch = 1; epoch <= EPOCHS; epoch++) {
        for(int b = 1; b <= batches; b++) {
            copy_batch_data<<<gridSize, blockSize>>>(device_X, device_Y, device_Xb, device_Yb, BATCH_SIZE, b - 1);
            cudaDeviceSynchronize();

            forward_propagation<<<gridSize, blockSize>>>(device_Xb, deviceW1, deviceW2, deviceB1, deviceB2, device_Y_hat, device_H);
            cudaDeviceSynchronize();

            calculate_error<<<gridSize, blockSize>>>(device_Yb, device_Y_hat, device_E);
            cudaDeviceSynchronize();

            transposeMatrices<<<gridSize, blockSize>>>(device_Xb, device_Xb_t, device_H, device_H_t, deviceW2, device_W2_t);
            cudaDeviceSynchronize();

            back_propagation<<<gridSize, blockSize>>>(device_Xb_t, device_Y_hat, device_H, device_H_t, device_E, deviceW1, deviceW2, device_W2_t, deviceB1, deviceB2);
            cudaDeviceSynchronize();
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float dtime = 0;
    cudaEventElapsedTime(&dtime, start, stop);

    printf("Training time: %fms\n", dtime);
}