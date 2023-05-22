#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "../datasets/dataset.h"
#include "../serial/mlp.h"

#define ETA 0.0001
#define EPOCHS 10

#define NUM_FEATURES 14 
#define HIDDEN_SIZE 7
#define OUTPUTS 1

#define BATCH_SIZE 128
#define GRID_SIZE 32

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
        C[(globalIdx * HIDDEN_SIZE) + i] = cuda_tanh_activation(sum[i] + bias[i]);
    }  
}

__device__ void calculate_output_layer_activation(
    double* A, 
    const double* weights, 
    const double* bias, 
    double* C
) {
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

    double sum = 0.0;

    for(int i = 0; i < HIDDEN_SIZE; i++) {
        sum += A[(globalIdx * HIDDEN_SIZE) + i] * weights[i];
    }

    C[globalIdx] = cuda_tanh_activation(sum + bias[0]);  
}

__device__ void calculate_one_minus_matrix_squared(double* A, double* B, int cols)
{
    int rowId = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 0; i < cols; i++)
    {
        B[rowId * cols + i] = 1.0 - (A[rowId * cols + i] * A[rowId * cols + i]);
    }
}

__device__ void hadamard_product(double *A, double *B, double *C, int cols)
{
    int rowId = threadIdx.x + blockIdx.x * blockDim.x;
    
    for (int i = 0; i < cols; i++)
    {
        C[rowId * cols + i] = A[rowId * cols + i] * B[rowId * cols + i];
    }
}

__device__ void matrix_multiply(double *A, double *B, double *C, int m, int n, int p)
{
    int rowId = threadIdx.x + blockIdx.x * blockDim.x;

    if (rowId < m) {
        for (int col = 0; col < p; col++) {
            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += A[rowId * n + k] * B[k * p + col];
            }

            C[rowId * p + col] = sum;
        }
    }
}

__device__ void transpose1DArray(const double* input, double* output, int rows, int cols)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 0; i < cols; i++)
    {
        output[i * rows + idx] = input[idx * cols + i];
    }
}

__device__ void columnSum(double* A, double* sums, int rows, int cols)
{
    int rowId = threadIdx.x + blockIdx.x * blockDim.x;

    if (rowId < cols) {
        double col_sum = 0.0;

        for (int i = 0; i < rows; i++) {
            col_sum += A[i * cols + rowId];
        }

        sums[rowId] = col_sum;
    }
}

__device__ void updateWeights(double* W, const double* Wg, int rows, int cols)
{
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (globalIdx < rows) {
        for(int i = 0; i < cols; i++) {
            W[globalIdx * cols + i] -= ETA * Wg[globalIdx * cols + i];
        } 
    }
}

__device__ void updateBias(double* b, const double* bg, int size)
{
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (globalIdx < size) {
        b[globalIdx] -= ETA * bg[globalIdx];
    }
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
    int rowId = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ double H_local [BATCH_SIZE * HIDDEN_SIZE];
    __shared__ double Y_hat_local [BATCH_SIZE];
    
    calculate_hidden_layer_activation(Xb, W1, b1, H_local);
    calculate_output_layer_activation(H_local, W2, b2, Y_hat_local);

    Y_hat_out[rowId] = Y_hat_local[rowId];

    for(int i = 0; i < HIDDEN_SIZE; i++) {
        H_out[rowId * HIDDEN_SIZE + i] = H_local[rowId * HIDDEN_SIZE + i];
    }
}

__global__ void calculate_error(double* Yb, double* Y_hat, double* E_out) {
    int rowId = threadIdx.x + blockIdx.x * blockDim.x;
    E_out[rowId] = Y_hat[rowId] - Yb[rowId];
}

__global__ void back_propagation(double* Xb, double* Y_hat, double* H, double* E, double* W1, double* W2, double* b1, double* b2) {
    __shared__ double W2_g_local [HIDDEN_SIZE * OUTPUTS];
    __shared__ double b2_g_local [OUTPUTS];

    __shared__ double W1_g_local [HIDDEN_SIZE * NUM_FEATURES];
    __shared__ double b1_g_local [HIDDEN_SIZE];

    __shared__ double He [BATCH_SIZE * HIDDEN_SIZE];
    __shared__ double He_hadamard [BATCH_SIZE * HIDDEN_SIZE];

    __shared__ double H_t [BATCH_SIZE * HIDDEN_SIZE];
    __shared__ double W2_t [NUM_FEATURES * HIDDEN_SIZE];
    __shared__ double Xb_t [NUM_FEATURES * BATCH_SIZE];

    __shared__ double H_squared_minus [BATCH_SIZE * HIDDEN_SIZE];
    __shared__ double Y_hat_squared_minus [BATCH_SIZE];

    __shared__ double E_Y_hat_hadamard [BATCH_SIZE];

    transpose1DArray(H, H_t, BATCH_SIZE, HIDDEN_SIZE);
    transpose1DArray(W2, W2_t, HIDDEN_SIZE, OUTPUTS);
    transpose1DArray(Xb, Xb_t, BATCH_SIZE, NUM_FEATURES);

    calculate_one_minus_matrix_squared(H, H_squared_minus, HIDDEN_SIZE);
    calculate_one_minus_matrix_squared(Y_hat, Y_hat_squared_minus, 1);

    hadamard_product(E, Y_hat_squared_minus, E_Y_hat_hadamard, 1);
    matrix_multiply(H_t, E_Y_hat_hadamard, W2_g_local, HIDDEN_SIZE, BATCH_SIZE, OUTPUTS);
    
    matrix_multiply(E_Y_hat_hadamard, W2_t, He, BATCH_SIZE, OUTPUTS, HIDDEN_SIZE);
    hadamard_product(He, H_squared_minus, He_hadamard, HIDDEN_SIZE);

    matrix_multiply(Xb_t, He_hadamard, W1_g_local, NUM_FEATURES, BATCH_SIZE, HIDDEN_SIZE);

    __syncthreads();

    columnSum(E_Y_hat_hadamard, b2_g_local, BATCH_SIZE, OUTPUTS);
    columnSum(He_hadamard, b1_g_local, BATCH_SIZE, HIDDEN_SIZE);

    __syncthreads();

    // Update weights & bias
    updateWeights(W1, W1_g_local, NUM_FEATURES, HIDDEN_SIZE);
    updateWeights(W2, W2_g_local, HIDDEN_SIZE, OUTPUTS);

    updateBias(b1, b1_g_local, HIDDEN_SIZE);
    updateBias(b2, b2_g_local, OUTPUTS);
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

    double *device_E;
    cudaMalloc((void **)&device_E, BATCH_SIZE * sizeof(double));

    int batches = BATCH_SIZE >= dataset.train_set.size ? 1 : ((dataset.train_set.size + BATCH_SIZE - 1) / BATCH_SIZE);
    dim3 blockSize(BATCH_SIZE / GRID_SIZE);
    dim3 gridSize(GRID_SIZE);

    printf("Batches: %d, Block size: %d, grid size: %d\n", batches, BATCH_SIZE / GRID_SIZE, GRID_SIZE);

    cudaEventRecord(start);

    for(int epoch = 1; epoch <= EPOCHS; epoch++) {
        for(int b = 1; b <= batches; b++) {
            printf("Computing for batch %d\n", b);

            copy_batch_data<<<gridSize, blockSize>>>(device_X, device_Y, device_Xb, device_Yb, BATCH_SIZE, b - 1);
            cudaDeviceSynchronize();

            forward_propagation<<<gridSize, blockSize>>>(device_Xb, deviceW1, deviceW2, deviceB1, deviceB2, device_Y_hat, device_H);
            cudaDeviceSynchronize();

            calculate_error<<<gridSize, blockSize>>>(device_Yb, device_Y_hat, device_E);
            cudaDeviceSynchronize();

            back_propagation<<<gridSize, blockSize>>>(device_X, device_Y_hat, device_H, device_E, deviceW1, deviceW2, deviceB1, deviceB2);
            cudaDeviceSynchronize();
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float dtime = 0;
    cudaEventElapsedTime(&dtime, start, stop);

    printf("Training time: %.4fms\n", dtime);


    /*
    double* Y_hat = (double*)calloc(BATCH_SIZE, sizeof(double));
    double* E = (double*)calloc(BATCH_SIZE, sizeof(double));
    double* H = (double*)calloc(BATCH_SIZE * HIDDEN_SIZE, sizeof(double));
    cudaMemcpy(Y_hat, device_Y_hat, BATCH_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(E, device_E, BATCH_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(H, device_H, BATCH_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);



    printf("E\n");
    for(int i = 0; i < BATCH_SIZE; i++) {
        printf("%f \n", E[i]);
    }

    printf("Y_hat\n");
    for(int i = 0; i < BATCH_SIZE; i++) {
        printf("%f \n", Y_hat[i]);
    }

    printf("\n\nH\n");
    for(int i = 0; i < BATCH_SIZE; i++) {
        for(int j = 0; j < HIDDEN_SIZE; j++) {
            printf("%f ", H[i * HIDDEN_SIZE + j]);
        }

        printf("\n");
    }

    double* Xb = (double*)calloc(NUM_FEATURES * BATCH_SIZE, sizeof(double));
    double* Yb = (double*)calloc(BATCH_SIZE, sizeof(double));

    cudaMemcpy(Xb, device_Xb, NUM_FEATURES * BATCH_SIZE *  sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Yb, device_Yb, BATCH_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    for(int i = 0; i < BATCH_SIZE; i++) {
        for(int j = 0; j < NUM_FEATURES; j++) {
            printf("%f ", Xb[i * NUM_FEATURES + j]);
        }

        printf("Y= %f\n", Yb[i]);
    }*/
}