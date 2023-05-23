#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp.h"

#include "../datasets/dataset.h"
#include "../serial/mlp.h"

#define ETA 0.00001
#define EPOCHS 10
#define BATCH_SIZE 64

#define NUM_FEATURES 14 
#define HIDDEN_SIZE 7
#define OUTPUTS 1

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

void matrix_multiply_flattened_openmp(double *A, double *B, double *C, int m, int n, int p, int num_of_threads){
    omp_set_num_threads(num_of_threads);    
    
    int i, j, k;    
    double sum;
    #pragma omp parallel for 
    for (i = 0; i < m; i++){
        for (j = 0; j < p; j++){
            sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (k = 0; k < n; k++){
                //sum += A[i][k] * B[k][j];
                sum += A[i*n + k] * B[k*n + j];
            }
            C[i*p + j] = sum;
        }
    }        
}

void matrix_multiply_openmp(double **A, double **B, double **C, int m, int n, int p, int num_of_threads){
    omp_set_num_threads(num_of_threads);    
    int i, j, k;
    double sum;
    
    #pragma omp parallel for 
    for (i = 0; i < m; i++){
        
        for (j = 0; j < p; j++){            
            sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (k = 0; k < n; k++){
                sum += A[i][k] * B[k][j];
            }             
            
            C[i][j] = sum;
        }
    }        
}

void add_bias_openmp(double **A, double *bias, int m, int p, int num_of_threads){
    omp_set_num_threads(num_of_threads);
    int i, j;
    #pragma omp parallel for schedule(dynamic, 1)
    for (i = 0; i < m; i++){        
        //#pragma omp parallel for schedule(dynamic, 1)
        for(int j = 0; j < p; j++){
            A[i][j] += bias[j];
        }
    }
}

void tanh_activate_matrix_openmp(double **A, int m, int p, int num_of_threads){
    omp_set_num_threads(num_of_threads);
    int i, j;
    #pragma omp parallel for schedule(dynamic, 1)
    for (i = 0; i < m; i++){        
        //#pragma omp parallel for schedule(dynamic, 1)
        for(int j = 0; j < p; j++){
            A[i][j] = tanh(A[i][j]);
        }
    }
}

void calculate_layer_activation_openmp(double **A, double **weights, double *bias, double **C, int m, int n, int p, int num_of_threads){            
    /*
    double start_time;
    double end_time;
    
    
    printf("calculate_layer_activation_openmp()\n");
    start_time = omp_get_wtime();
    matrix_multiply_openmp(A, weights,C, m,n,p,num_of_threads);
    
    end_time = omp_get_wtime();
    printf("matrix_multiply_openmp(), num_of_threads: %d time: %.4fs\n", num_of_threads, end_time - start_time);
    
    double *flatA = flattenDoubleArray(A, m,n);        
    double *flatweights = flattenDoubleArray(weights, n, p);    
    double *flatC = flattenDoubleArray(C, m,p);
    
    start_time = omp_get_wtime();
    matrix_multiply_flattened_openmp(flatA, flatweights,flatC, m,n,p,num_of_threads);
    end_time = omp_get_wtime();
    printf("matrix_multiply_flattened_openmp(), num_of_threads: %d time: %.4fs\n", num_of_threads, end_time - start_time);
    exit(0);
    
    start_time = omp_get_wtime();
    add_bias_openmp(C,bias, m,p, num_of_threads);
    end_time = omp_get_wtime();
    printf("add_bias_openmp(), num_of_threads: %d time: %.4fs\n", num_of_threads, end_time - start_time);
    
    start_time = omp_get_wtime();
    tanh_activate_matrix_openmp(C,m,p, num_of_threads);
    end_time = omp_get_wtime();
    printf("tanh_activate_matrix_openmp(), num_of_threads: %d time: %.4fs\n", num_of_threads, end_time - start_time);
    */
    
    matrix_multiply_openmp(A, weights,C, m,n,p,num_of_threads);
    add_bias_openmp(C,bias, m,p, num_of_threads);
    tanh_activate_matrix_openmp(C,m,p, num_of_threads);
    
}


void calcaulate_error_openmp(double *Y, double **Y_hat, double **E, int m, int num_of_threads){
    omp_set_num_threads(num_of_threads);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < m; i++){
        E[i][0] = Y[i] - Y_hat[i][0];
    }
}

void transpose_openmp(double **A, double **B, int rows, int cols, int num_of_threads){
    omp_set_num_threads(num_of_threads);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < rows; i++){
        //#pragma omp parallel for schedule(dynamic, 1)
        for (int j = 0; j < cols; j++){
            B[j][i] = A[i][j];
        }
    }
}

void square_openmp(double **A, int rows, int cols, int num_of_threads){
    omp_set_num_threads(num_of_threads);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < rows; i++){
        //#pragma omp parallel for schedule(dynamic, 1)
        for (int j = 0; j < cols; j++){
            A[i][j] = A[i][j] * A[i][j];
        }
    }
}

void calculate_one_minus_matrix_openmp(double **A, double **B, int rows, int cols, int num_of_threads){
    omp_set_num_threads(num_of_threads);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < rows; i++){
        //#pragma omp parallel for schedule(dynamic, 1)
        for (int j = 0; j < cols; j++){
            B[i][j] = 1 - A[i][j];
        }
    }
}

void hadamard_product_openmp(double **A, double **B, double **C, int rows, int cols, int num_of_threads){
    omp_set_num_threads(num_of_threads);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < rows; i++){
        //#pragma omp parallel for schedule(dynamic, 1)
        for (int j = 0; j < cols; j++){
            C[i][j] = A[i][j] * B[i][j];
        }
    }
}

void column_sum_openmp(double **A, double *sums, int rows, int cols, int num_of_threads){
    omp_set_num_threads(num_of_threads);
    #pragma omp parallel for 
    for (int j = 0; j < cols; j++)
    {
        double col_sum = 0.0;
        #pragma omp parallel for reduction(+:col_sum)
        for (int i = 0; i < rows; i++)
        {
            col_sum += A[i][j];
        }

        sums[j] = col_sum;
    }
}

void update_weights_openmp(double **W, double **Wg, float eta, int rows, int cols, int num_of_threads){
    omp_set_num_threads(num_of_threads);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < rows; i++){
        //#pragma omp parallel for schedule(dynamic, 1)
        for (int j = 0; j < cols; j++){
            W[i][j] -= eta * Wg[i][j];
        }
    }    
}

void update_bias_openmp(double *b, double *bg, float eta, int size, int num_of_threads){
    omp_set_num_threads(num_of_threads);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < size; i++){
        b[i] -= eta * bg[i];
    }
}


void forward_propagation_openmp(double **H, double **Xb, MLP *mlp, double **Y_hat, int batchRows, int num_of_threads){
    //printf("\n\n\nforward_propagation_openmp()\n");
    double start_time = omp_get_wtime();
    calculate_layer_activation_openmp(Xb,
        mlp->hiddenLayer.W1,
        mlp->hiddenLayer.b1,
        H,
        batchRows,
        mlp->inputLayer.features,
        mlp->hiddenLayer.nodes,
        num_of_threads);

     calculate_layer_activation_openmp(H,
        mlp->outputLayer.W2,
        mlp->outputLayer.b2,
        Y_hat,
        batchRows,
        mlp->hiddenLayer.nodes,
        mlp->outputLayer.nodes,
        num_of_threads);
    double end_time = omp_get_wtime();

    //printf("forward_propagation_openmp(), num_of_threads: %d time: %.4fs\n", num_of_threads, end_time - start_time);
    

}
void free_2d_double_openmp(double **array, int rows, int num_of_threads){
    //printf("num_of_threads: %d\n", num_of_threads);    
    omp_set_num_threads(num_of_threads);
    /*
    #pragma omp parallel for
    for (int i = 0; i < rows; i++){
        free(array[i]);
    }
    
    free(array);
    */
}

void back_propagation_openmp(MLP *mlp, double **Xb, double **H, double *Y, double **Y_hat, double **E, int batchRows, double eta, int num_of_threads){
    // Initialize matrices
    
    double **H_transpose = (double **)malloc(mlp->hiddenLayer.nodes * sizeof(double *));
    double **Xb_transpose = (double **)malloc(mlp->inputLayer.features * sizeof(double *));
    double **W2_transpose = (double **)malloc(mlp->outputLayer.nodes * sizeof(double *));

    double **Y_hat_squared_minus_one = (double **)malloc(batchRows * sizeof(double *));
    double **H_squared_minus_one = (double **)malloc(batchRows * sizeof(double *));

    double **E_Y_hat_hadamard = (double **)malloc(batchRows * sizeof(double *));

    double **W2_g = (double **)malloc(mlp->hiddenLayer.nodes * sizeof(double *));
    double *b2_g = (double *)malloc(mlp->outputLayer.nodes * sizeof(double));

    double **W1_g = (double **)malloc(mlp->inputLayer.features * sizeof(double *));
    double *b1_g = (double *)malloc(mlp->hiddenLayer.nodes * sizeof(double));

    double **He = (double **)malloc(batchRows * sizeof(double *));
    double **He_hadamard = (double **)malloc(batchRows * sizeof(double *));

    for (int i = 0; i < mlp->hiddenLayer.nodes; i++){
        H_transpose[i] = (double *)malloc(batchRows * sizeof(double));
        W2_g[i] = (double *)malloc(mlp->outputLayer.nodes * sizeof(double));
    }

    for (int i = 0; i < mlp->inputLayer.features; i++){
        W1_g[i] = (double *)malloc(mlp->hiddenLayer.nodes * sizeof(double));
        Xb_transpose[i] = (double *)malloc(batchRows * sizeof(double));
    }

    for (int i = 0; i < mlp->outputLayer.nodes; i++){
        W2_transpose[i] = (double *)malloc(mlp->hiddenLayer.nodes * sizeof(double));
    }

    for (int i = 0; i < batchRows; i++){
        Y_hat_squared_minus_one[i] = (double *)malloc(mlp->outputLayer.nodes * sizeof(double));
        H_squared_minus_one[i] = (double *)malloc(mlp->hiddenLayer.nodes * sizeof(double));

        E_Y_hat_hadamard[i] = (double *)malloc(mlp->outputLayer.nodes * sizeof(double));
        He[i] = (double *)malloc(mlp->hiddenLayer.nodes * sizeof(double));
        He_hadamard[i] = (double *)malloc(mlp->hiddenLayer.nodes * sizeof(double));
    }
    
    calcaulate_error_openmp(Y, Y_hat, E, batchRows, num_of_threads);

    // Transpose H to get H_transpose values
    transpose_openmp(H, H_transpose, batchRows, mlp->hiddenLayer.nodes, num_of_threads);

    // Transpose W2 to get W2_transpoe
    transpose_openmp(mlp->outputLayer.W2, W2_transpose, mlp->hiddenLayer.nodes, mlp->outputLayer.nodes, num_of_threads);

    // Transpose Xb to get Xb_transpose values
    transpose_openmp(Xb, Xb_transpose, batchRows, mlp->inputLayer.features, num_of_threads);

    // Y_hat squared
    square_openmp(Y_hat, batchRows, 1, num_of_threads);

    // H squared
    square_openmp(H, batchRows, mlp->hiddenLayer.nodes, num_of_threads);
    
    // printf("\n");
    //  (1 - Y_hat squared) & (1 - H squared)
    calculate_one_minus_matrix_openmp(Y_hat, Y_hat_squared_minus_one, batchRows, mlp->outputLayer.nodes, num_of_threads);
    calculate_one_minus_matrix_openmp(H, H_squared_minus_one, batchRows, mlp->hiddenLayer.nodes, num_of_threads);

    // Calculate W2_g -> H_transpose(E ◦ (1 − Yˆ^◦2))
    hadamard_product_openmp(E, Y_hat_squared_minus_one, E_Y_hat_hadamard, batchRows, mlp->outputLayer.nodes, num_of_threads);
    matrix_multiply_openmp(H_transpose, E_Y_hat_hadamard, W2_g, mlp->hiddenLayer.nodes, batchRows, mlp->outputLayer.nodes, num_of_threads);

    // Calculate b2_g -> 1_transpose(E ◦ (1 − Yˆ^◦2))
    column_sum_openmp(E_Y_hat_hadamard, b2_g, batchRows, mlp->outputLayer.nodes, num_of_threads);

    // Calculate He -> (E◦(1−Yˆ◦2))W2_transpose
    // He ← He ◦ (1 − H◦2)
    matrix_multiply_openmp(E_Y_hat_hadamard, W2_transpose, He, batchRows, mlp->outputLayer.nodes, mlp->hiddenLayer.nodes, num_of_threads);
    hadamard_product_openmp(He, H_squared_minus_one, He_hadamard, batchRows, mlp->hiddenLayer.nodes, num_of_threads);

    // Calculate W1_g ← Xb_transpose He
    matrix_multiply_openmp(Xb_transpose, He_hadamard, W1_g, mlp->inputLayer.features, batchRows, mlp->hiddenLayer.nodes, num_of_threads);

    // Calculate b1_g -> 1_transpose He
    column_sum_openmp(He_hadamard, b1_g, batchRows, mlp->hiddenLayer.nodes, num_of_threads);

    // Update weights & bias
    update_weights_openmp(mlp->hiddenLayer.W1, W1_g, eta, mlp->inputLayer.features, mlp->hiddenLayer.nodes, num_of_threads);
    update_weights_openmp(mlp->outputLayer.W2, W2_g, eta, mlp->hiddenLayer.nodes, mlp->outputLayer.nodes, num_of_threads);

    update_bias_openmp(mlp->hiddenLayer.b1, b1_g, eta, mlp->hiddenLayer.nodes, num_of_threads);
    update_bias_openmp(mlp->outputLayer.b2, b2_g, eta, mlp->outputLayer.nodes, num_of_threads);
    /*
    free_2d_double_openmp(H_transpose, mlp->hiddenLayer.nodes, num_of_threads);    
    */
    free_2d_double_openmp(W2_transpose, mlp->outputLayer.nodes, num_of_threads);    
    free_2d_double_openmp(Xb_transpose, mlp->inputLayer.features, num_of_threads);    
    free_2d_double_openmp(W2_g, mlp->hiddenLayer.nodes, num_of_threads);
    free_2d_double_openmp(W1_g, mlp->inputLayer.features, num_of_threads);
    
    free_2d_double_openmp(Y_hat_squared_minus_one, batchRows, num_of_threads);
    free_2d_double_openmp(He, batchRows, num_of_threads);
    free_2d_double_openmp(H_squared_minus_one, batchRows, num_of_threads);
    free_2d_double_openmp(E_Y_hat_hadamard, batchRows, num_of_threads);
    
    free(b1_g);
    free(b2_g);
    
    /*
    */
}

void train_mlp_openmp(MLP *mlp, double *Y, float eta, int epochs, int batchSize, Data * test_set, int num_of_threads){    
    printf("\n\n\ntrain_mlp_openmp(): \n");
    
    int batches = batchSize >= mlp->inputLayer.samples ? 1 : (mlp->inputLayer.samples / batchSize);
    printf("ntrain_mlp_openmp(): Batches num: %d\n", batches);
    // allocate memory for intermediate matrices and vectors
    double **Xb = (double **)malloc(batchSize * sizeof(double *));
    double **Yb = (double **)malloc(batchSize * sizeof(double *));

    double **H = (double **)malloc(batchSize * sizeof(double *));

    double **Y_hat = (double **)malloc(batchSize * sizeof(double *));

    double **E = (double **)malloc(batchSize * sizeof(double *));
    
    
    for (int i = 0; i < batchSize; i++)
    {
        Xb[i] = (double *)malloc(mlp->inputLayer.features * sizeof(double));
        Yb[i] = (double *)malloc(sizeof(double));

        H[i] = (double *)malloc(mlp->hiddenLayer.nodes * sizeof(double));
        Y_hat[i] = (double *)malloc(sizeof(double));

        E[i] = (double *)malloc(sizeof(double));
    }
    
    printf("train_mlp_openmp(): matrices initialized\n");
    test_mlp(mlp, test_set);
    printf("\n");
    
    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        
        printf("Epoch: %d\n", epoch);
        for (int b = 1; b <= batches; b++)
        {
            
            int batchStart = (b - 1) * batchSize;
            int batchEnd = b * batchSize - 1;

            if (batchEnd >= mlp->inputLayer.samples)
            {
                batchEnd = mlp->inputLayer.samples - 1;
            }
            copy_batch_data(Xb, Yb, Y, mlp, batchStart, batchEnd);
            
            forward_propagation_openmp(H, Xb, mlp, Y_hat, batchSize, num_of_threads);
                 
            back_propagation_openmp(mlp, Xb, H, Y, Y_hat, E, batchSize, eta, num_of_threads);
            
            /*
            */    
        }        
        test_mlp(mlp, test_set);
        printf("\n");
        
    }
    
    free_2d_double_openmp(Xb, batchSize, num_of_threads);
    free_2d_double_openmp(Yb, batchSize, num_of_threads);
    free_2d_double_openmp(H, batchSize, num_of_threads);
    free_2d_double_openmp(Y_hat, batchSize, num_of_threads);
    free_2d_double_openmp(E, batchSize, num_of_threads);
    printf("train_mlp_openmp(): done\n\n\n");
    
}
int main(int argc, char *argv[]){
    DATASET dataset;
    MLP mlp;
    int num_of_threads;    
    
    num_of_threads = atoi(argv[1]);
    
    prepare_dataset("../datasets/adults/encoded_data_normalized.data", NUM_FEATURES, OUTPUTS, &dataset);
    init_mlp(&mlp, dataset.train_set.X, dataset.train_set.size, NUM_FEATURES, OUTPUTS, HIDDEN_SIZE);
    // print_mlp_info(&mlp, 10);
    
    printf("Start training...\n");
    
    double start_time = omp_get_wtime();
    
    train_mlp_openmp(&mlp, dataset.train_set.Y, ETA, EPOCHS, BATCH_SIZE, &dataset.test_set, num_of_threads);
    
    double end_time = omp_get_wtime();

    printf("Training complete, time: %.4fs\n", end_time - start_time);

    printf("Start testing...\n");
    test_mlp(&mlp, &dataset.test_set);
    /*
    clear_mlp(&mlp, HIDDEN_SIZE);
    clear_dataset(&dataset);
    */
    
    return 0;
}

