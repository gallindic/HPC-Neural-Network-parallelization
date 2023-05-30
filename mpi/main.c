// module load OpenMPI/4.1.0-GCC-10.2.0 
// mpicc -o hello hello.c
// srun --reservation=fri --mpi=pmix --nodes=2 ./hello

#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>
#include "omp.h"
#include "math.h"
#include "../datasets/dataset.h"
#include "../serial/mlp.h"

#define MASTER 0
#define EPOCHS 1
#define HIDDEN_SIZE 7
#define OUTPUTS 1
#define ETA 0.0001
#define NUM_FEATURES 14

void print_matrix_int(int ** M,int rows, int cols, char * str){
    printf("\n%s:\n", str);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3d ", M[i][j]);
        }
        printf("\n");
    }
}

void print_matrix_double(double ** M,int rows, int cols, char * str){
    printf("\n%s:\n", str);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3.3f ", M[i][j]);
        }
        printf("\n");
    }
}
int **alloc_2d_int(int rows, int cols) {
    int *data = (int *)malloc(rows*cols*sizeof(int));
    int **array= (int **)malloc(rows*sizeof(int*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);

    return array;
}

double **alloc_2d_double(int rows, int cols) {
    double *data = (double *)malloc(rows*cols*sizeof(double));
    double **array= (double **)malloc(rows*sizeof(double*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);

    return array;
}

void distribute_matrix(double ***M, int *rows, int cols, int * offset_rows, int myid, int procs){
    MPI_Status  status;
    
    int tag = 0;
    int rows_per_process = rows[0] / procs;
    int remaining_rows = rows[0] % procs;
    int start_row, end_row, srows;
    int * sendcounts;
    int * displs;
    
    sendcounts = (int *)malloc(procs*sizeof(int));
    displs = (int *)malloc(procs*sizeof(int));

    for(int i = 0; i < procs; i++){
        start_row = i*(rows_per_process*cols);
        end_row = (i+1)*(rows_per_process*cols); 
        if(remaining_rows > 0){
            end_row += cols;
            remaining_rows--;
        }

        srows = end_row - start_row;                
        sendcounts[i] = srows;
        if(i == 0){
            displs[i] = 0;
        } else {
            displs[i] = displs[i-1] + sendcounts[i-1];
        }    
    }

    int recvbuf_length = sendcounts[myid];
    *offset_rows = displs[myid] / rows[0];
    *rows = recvbuf_length / cols;
    
    double ** m = alloc_2d_double(recvbuf_length, cols);
    
    if(myid == MASTER){
        MPI_Scatterv(&M[0][0][0], sendcounts, displs, MPI_DOUBLE, &m[0][0], recvbuf_length, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);        
    } else {
        MPI_Scatterv(NULL, sendcounts, displs, MPI_DOUBLE, &m[0][0], recvbuf_length, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);        
    }
        
    *M = m;        
    //return myrows;
}

double ** _matmul_(double **M, double **N, int rowsM, int colsM, int colsN){
    // colsM == rowsN
    int i,j,k;
    double sum;
    double **R = alloc_2d_double(rowsM, colsN);
    for(i = 0; i < rowsM; i++){
        for(j = 0; j < colsN; j++){
            sum = 0;
            for(k = 0; k < colsM; k++){                
                sum += M[i][k] * N[k][j];
            }            
            R[i][j] = sum;
        }
    }
    return R;
}

double ** matmul_mpi(double** M, double ** N, int rowsM, int colsM, int colsN, int myid, int procs){
    
    double ** finalR;
    
    if(myid == MASTER){
        finalR = alloc_2d_double(rowsM, colsN);
    } else {
        finalR = NULL;
    } 
    

    MPI_Bcast(&N[0][0], colsM*colsN, MPI_DOUBLE, 0, MPI_COMM_WORLD);    
    int offset_rows;
    distribute_matrix(&M, &rowsM,colsM,&offset_rows, myid,procs);    

    double ** R = _matmul_(M,N, rowsM, colsM, colsN);
    
    MPI_Barrier(MPI_COMM_WORLD);
               
    if(myid == MASTER){
        MPI_Gather(&R[0][0], rowsM*colsN, MPI_DOUBLE, &finalR[0][0], rowsM*colsN, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    } else {
        MPI_Gather(&R[0][0], rowsM*colsN, MPI_DOUBLE, NULL, rowsM*colsN, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    }
    
    return finalR;
}

double ** _add_bias_transposed_(double **M, double *bias, int rows, int cols, int offset_rows){
    double **R = alloc_2d_double(rows, cols);
    for(int j = 0; j < cols; j++){
        for (int i = 0; i < rows; i++){                     
            R[i][j] = M[i][j] + bias[offset_rows + i];
        }
    }
    return R;
}

double ** add_bias_mpi(double **M, double *bias, int rows, int cols, int myid, int procs){
    double ** finalR;
    if(myid == MASTER){
        finalR = alloc_2d_double(rows, cols);
    } else {
        finalR = NULL;
    } 
    int offset_rows;
    
    MPI_Bcast(&bias[0], cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);    
    distribute_matrix(&M, &rows,cols,&offset_rows, myid,procs);    
        
    double **R = _add_bias_transposed_(M, bias, rows, cols, offset_rows);    
    
    MPI_Barrier(MPI_COMM_WORLD);    
    
    if(myid == MASTER){
        MPI_Gather(&R[0][0], rows*cols, MPI_DOUBLE, &finalR[0][0], rows*cols, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    } else {
        MPI_Gather(&R[0][0], rows*cols, MPI_DOUBLE, NULL, rows*cols, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    }
    
    return finalR;

}

double ** _haddamard_product_(double **M, double **N, int rows, int cols){
    int i,j,k;
    double sum;
    double **R = alloc_2d_double(rows, cols);
    for(i = 0; i < rows; i++){
        for(j = 0; j < cols; j++){                                                
            R[i][j] = M[i][j] + N[i][j];
        }
    }
    return R;
}

double ** haddamard_product_mpi(double** M, double ** N, int rows, int cols, int myid, int procs){
    
    double ** finalR;
    if(myid == MASTER){
        finalR = alloc_2d_double(rows, cols);
    } else {
        finalR = NULL;
    } 
    
    int offset_rows;

    int rowsM = rows;
    int colsM = cols;
    int rowsN = rows;
    int colsN = cols;
    
    distribute_matrix(&M, &rowsM,colsM,&offset_rows, myid,procs);    
    distribute_matrix(&N, &rowsN,colsN,&offset_rows, myid,procs);  
    
    double ** R = _haddamard_product_(M,N, rowsM, colsM);
    
    
    MPI_Barrier(MPI_COMM_WORLD);
               
    if(myid == MASTER){
        MPI_Gather(&R[0][0], rowsM*colsM, MPI_DOUBLE, &finalR[0][0], rowsM*colsM, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    } else {
        MPI_Gather(&R[0][0], rowsM*colsM, MPI_DOUBLE, NULL, rowsM*colsM, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    }
    
    return finalR;

}

double ** _tanh_activate_(double **M, int rows, int cols){
    int i,j,k;
    double sum;
    double **R = alloc_2d_double(rows, cols);
    for(i = 0; i < rows; i++){
        for(j = 0; j < cols; j++){                                                
            R[i][j] = tanh(M[i][j]);
        }
    }
    return R;
}

double ** tanh_activate_mpi(double** M, int rows, int cols, int myid, int procs){
    
    double ** finalR;
    if(myid == MASTER){
        finalR = alloc_2d_double(rows, cols);
    } else {
        finalR = NULL;
    } 
    int offset_rows;
        
    distribute_matrix(&M, &rows,cols,&offset_rows, myid,procs);    
        
    double **R = _tanh_activate_(M, rows, cols);    
    
    MPI_Barrier(MPI_COMM_WORLD);    
    
    if(myid == MASTER){
        MPI_Gather(&R[0][0], rows*cols, MPI_DOUBLE, &finalR[0][0], rows*cols, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    } else {
        MPI_Gather(&R[0][0], rows*cols, MPI_DOUBLE, NULL, rows*cols, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    }
    
    return finalR;

}

double ** calculate_layer_activation_mpi(double **A, double **weights, double *bias, double **C, int m, int n, int p, int myid, int procs){            
    C = matmul_mpi(A, weights, m,n,p, myid, procs);
    MPI_Barrier(MPI_COMM_WORLD);

    C = add_bias_mpi(C,bias, m,p, myid, procs);
    MPI_Barrier(MPI_COMM_WORLD);

    C = tanh_activate_mpi(C,m,p, myid, procs);
    MPI_Barrier(MPI_COMM_WORLD);
    return C;
}

double ** _calculate_error_(double **Y, double **Y_hat, int rows, int cols){
    int i,j,k;
    double **E = alloc_2d_double(rows, cols);
    for(i = 0; i < rows; i++){
        for(j = 0; j < cols; j++){                                                
            E[i][j] = Y[i][j] - Y_hat[i][j];
        }
    }
    return E;
}

double ** calculate_error_mpi(double **Y, double **Y_hat, int rows, int cols, int myid, int procs) {
   double ** finalE;
    if(myid == MASTER){
        finalE = alloc_2d_double(rows, cols);
    } else {
        finalE = NULL;
    } 
    
    int offset_rows;
    
    distribute_matrix(&Y, &rows, cols, &offset_rows, myid, procs);    
    distribute_matrix(&Y_hat, &rows, cols, &offset_rows, myid, procs);  
    
    double ** R = _calculate_error_(Y, Y_hat, rows, cols);
    
    MPI_Barrier(MPI_COMM_WORLD);
               
    if(myid == MASTER){
        MPI_Gather(&R[0][0], rows*cols, MPI_DOUBLE, &finalE[0][0], rows*cols, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    } else {
        MPI_Gather(&R[0][0], rows*cols, MPI_DOUBLE, NULL, rows*cols, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    }
    
    return finalE;
}

double ** _transpose_matrix_(double **A, int rows, int cols, int myid){
    double **A_transposed = alloc_2d_double(cols, rows);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){   
            A_transposed[j][i] = A[i][j];
        }
    }
    printf("Rank: %d, rows: %d, cols: %d", myid, rows, cols);
    print_matrix_double(A_transposed, cols, rows, "A_transposed");
    return A_transposed;
}

double ** transpose_mpi(double **A, int rows, int cols, int myid, int procs) {
   double ** finalTransposed;
    if(myid == MASTER){
        printf("Global dimenssions: %d rows, %d cols\n", rows, cols);
        finalTransposed = alloc_2d_double(cols, rows);
    } else {
        finalTransposed = NULL;
    } 
    
    int offset_rows;
    
    distribute_matrix(&A, &rows, cols, &offset_rows, myid, procs);    
    
    double ** A_transposed = _transpose_matrix_(A, rows, cols, myid);
    
    MPI_Barrier(MPI_COMM_WORLD);
               
    if(myid == MASTER){
        MPI_Gather(&A_transposed[0][0], rows*cols, MPI_DOUBLE, &finalTransposed[0][0], rows*cols, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    } else {
        MPI_Gather(&A_transposed[0][0], rows*cols, MPI_DOUBLE, NULL, rows*cols, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    }
    
    return finalTransposed;
}

double ** _square_matrix_(double **A, int rows, int cols){
    double **A_squared = alloc_2d_double(rows, cols);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){                                               
            A_squared[i][j] = A[i][j] * A[i][j];
        }
    }
    return A_squared;
}

double ** square_mpi(double** A, int rows, int cols, int myid, int procs){
    
    double ** finalSquared;
    if(myid == MASTER){
        finalSquared = alloc_2d_double(rows, cols);
    } else {
        finalSquared = NULL;
    } 
    int offset_rows;
        
    distribute_matrix(&A, &rows, cols, &offset_rows, myid, procs);    
        
    double **squared = _square_matrix_(A, rows, cols);    
    
    MPI_Barrier(MPI_COMM_WORLD);    
    
    if(myid == MASTER){
        MPI_Gather(&squared[0][0], rows*cols, MPI_DOUBLE, &finalSquared[0][0], rows*cols, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    } else {
        MPI_Gather(&squared[0][0], rows*cols, MPI_DOUBLE, NULL, rows*cols, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    }
    
    return finalSquared;

}

double ** _one_minus_matrix_(double **A, int rows, int cols){
    double **one_minus_matrix = alloc_2d_double(rows, cols);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){                                               
            one_minus_matrix[i][j] = 1 - A[i][j];
        }
    }
    return one_minus_matrix;
}

double ** calculate_one_minus_matrix_mpi(double** A, int rows, int cols, int myid, int procs){
    
    double ** finalMatrix;
    if(myid == MASTER){
        finalMatrix = alloc_2d_double(rows, cols);
    } else {
        finalMatrix = NULL;
    } 
    int offset_rows;
        
    distribute_matrix(&A, &rows, cols, &offset_rows, myid, procs);    
        
    double **matrix = _one_minus_matrix_(A, rows, cols);    
    
    MPI_Barrier(MPI_COMM_WORLD);    
    
    if(myid == MASTER){
        MPI_Gather(&matrix[0][0], rows*cols, MPI_DOUBLE, &finalMatrix[0][0], rows*cols, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    } else {
        MPI_Gather(&matrix[0][0], rows*cols, MPI_DOUBLE, NULL, rows*cols, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    }
    
    return finalMatrix;

}


// int main(int argc, char* argv[])
// {       
    
// 	int myid, procs;        
//     int rowsM = 8;
//     int colsM = 8;
//     int colsN = 8;
//     double **M;    
//     double **N;    
//     double *bias;    
//     int tag = 0;
//     MPI_Status  status;
    
// 	MPI_Init(&argc, &argv); // initialize MPI 
// 	MPI_Comm_rank(MPI_COMM_WORLD, &myid);	
// 	MPI_Comm_size(MPI_COMM_WORLD, &procs);	
//     N = alloc_2d_double(colsM, colsN);
    
//     char nodename[MPI_MAX_PROCESSOR_NAME];
//     int nodename_len;
//     MPI_Get_processor_name(nodename, &nodename_len);
//     //printf("nodename: %s\n", nodename);
//     if(myid == MASTER){
//         M = alloc_2d_double(rowsM, colsM);
//         bias = (double *)malloc(colsM*sizeof(double));
//         for(int i = 0; i < rowsM; i++){            
//             for(int j = 0; j < colsM; j++){
//                 M[i][j] = (double) i;                
//                 bias[j] = (double) j;
//             }            
//         }
        
//         for(int i = 0; i < colsM; i++){            
//             for(int j = 0; j < colsN; j++){                
//                 N[i][j] = (double) i;                
//             }
//         }
//         print_matrix_double(M, rowsM, colsM,"M");
//         // printf("\nbias: \n");
//         // for(int i = 0; i < colsM; i++){
//         //    printf("%f\n", bias[i]);
//         // }
//         // print_matrix_double(M, colsM, colsN, "N");
//     }

//     double start_time = MPI_Wtime();
//     //M = haddamard_product_mpi(M,N, rows, cols, myid, procs);
//     //M = matmul_mpi(M,N, rowsM, colsM, colsN, myid, procs);
//     //M = add_bias_mpi(M,bias, rows, cols, myid, procs);
//     //M = tanh_activate_mpi(M, rows, cols, myid, procs);


//     //** Forward propagation
//     // H <- tanh(X_b * W_1 + b_1)
//     double **H;
//     H = calculate_layer_activation_mpi(M, N, bias, H, rowsM, colsM, colsN, myid, procs);
//     // Y_hat <- tanh(H * W_2 + b_2)
//     double **Y_hat;
//     Y_hat = calculate_layer_activation_mpi(M, N, bias, Y_hat, rowsM, colsM, colsN, myid, procs);


//     //** Back propagation
//     // E <- Y_hat + Y_b
//     double **E;
//     E = calculate_error_mpi(H, Y_hat, rowsM, colsM, myid, procs);
//     // W_2g <- H_t * (E ° (1 - Y_hat_squared))
//     // Transposed
//     //double **M_transposed;
//     //M_transposed = transpose_mpi(M, rowsM, colsM, myid, procs); // Transpose H
//     //M_transposed = _transpose_matrix_(M, rowsM, colsM); // Transpose H
//     // Squared
//     double **M_squared;
//     M_squared = square_mpi(M, rowsM, colsM, myid, procs);
//     // One minus matrix
//     double **one_minus_M;
//     one_minus_M = calculate_one_minus_matrix_mpi(M, rowsM, colsM, myid, procs);

     
//     MPI_Barrier(MPI_COMM_WORLD);
//     if(myid == MASTER) {
//         //print_matrix_double(H, colsM, colsN, "H");
//         //print_matrix_double(M_transposed, colsM, rowsM, "M_transposed");
//         //print_matrix_double(E, colsM, colsN, "E");
//         print_matrix_double(one_minus_M, rowsM, colsM, "One minus M");
//     }
//     if(myid == MASTER){
//         double end_time = MPI_Wtime();
//         printf("time = %f\n", end_time-start_time);
//     }
//     //printf("myid = %d | rows = %d | cols = %d\n", myid, rows, cols);
//     //if(myid == MASTER){
//     //    //printf("myid = %d | rows = %d | cols = %d\n", myid, rows, cols);
//     //    print_matrix_double(C,rows, cols,"C final");
//     //}

//     // if(myid == MASTER){
//     //    free_2d_double(M, rowsM);
//     //    free_2d_double(N, rowsM);
//     //    free_2d_double(H, rowsM);
//     //    free_2d_double(Y_hat, rowsM);
//     //    free_2d_double(E, rowsM);
//     //    free_2d_double(H_transposed, rowsM);
//     // } 
// 	MPI_Finalize();
    
// 	return 0;
// }

void train_mlp_mpi(MLP* mlp, double* Y, float eta, int epochs, int batchSize, Data* test_set, int myid, int procs) {
    int batches = batchSize >= mlp->inputLayer.samples ? 1 : (mlp->inputLayer.samples / batchSize);
    printf("train_mlp_mpi(): Batches num: %d\n", batches);

    if (myid == MASTER) {
        // Allocate memory for intermediate matrices and vectors
        double** Xb = (double**)malloc(batchSize * sizeof(double*));
        double** Yb = (double**)malloc(batchSize * sizeof(double*));

        for (int i = 0; i < batchSize; i++) {
            Xb[i] = (double*)malloc(mlp->inputLayer.features * sizeof(double));
            Yb[i] = (double*)malloc(sizeof(double));
        }

        printf("train_mlp_mpi(): matrices initialized\n");
        test_mlp(mlp, test_set);
        printf("\n");
    }

    for (int epoch = 1; epoch <= epochs; epoch++) {
        printf("Epoch: %d\n", epoch);
        for (int b = 1; b <= batches; b++) {
            printf("Batch: %d\n", b);

            int batchStart = (b - 1) * batchSize;
            int batchEnd = b * batchSize - 1;

            if (batchEnd >= mlp->inputLayer.samples) {
                batchEnd = mlp->inputLayer.samples - 1;
            }
            copy_batch_data(Xb, Yb, Y, mlp, batchStart, batchEnd);

            // Forward propagation
            double** H;
            if (myid == MASTER) {
                printf("Xb:\n");
                print_matrix_double(Xb, batchSize, NUM_FEATURES, "Xb");
                printf("W1:\n");
                print_matrix_double(mlp->hiddenLayer.W1, NUM_FEATURES, HIDDEN_SIZE, "W1");
                printf("\nb1:\n");
                for (int i = 0; i < HIDDEN_SIZE; i++) {
                    printf("%f\n", mlp->hiddenLayer.b1[i]);
                }
            }

            H = calculate_layer_activation_mpi(Xb,
                mlp->hiddenLayer.W1,
                mlp->hiddenLayer.b1,
                H,
                batchSize,
                mlp->inputLayer.features,
                mlp->hiddenLayer.nodes,
                myid,
                procs);

            double** Y_hat;
            Y_hat = calculate_layer_activation_mpi(H,
                mlp->outputLayer.W2,
                mlp->outputLayer.b2,
                Y_hat,
                batchSize,
                mlp->hiddenLayer.nodes,
                mlp->outputLayer.nodes,
                myid,
                procs);
        }
        if (myid == MASTER) {
            test_mlp(mlp, test_set);
            printf("\n");
        }
    }
}

int main(int argc, char** argv) {
    DATASET dataset;
    MLP mlp;

    int myid, procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    int batchSize = procs;
    int rowsPerProcess, batches, remainingRows;

    // Prepare dataset in the root process
    if(myid == MASTER) {
        prepare_dataset("../datasets/adults/encoded_data_small.data", NUM_FEATURES, OUTPUTS, &dataset);  
        init_mlp(&mlp, dataset.train_set.X, dataset.train_set.size, NUM_FEATURES, OUTPUTS, HIDDEN_SIZE);
        //print_matrix_double(dataset.train_set.X, dataset.train_set.size, NUM_FEATURES, "Dataset");
        printf("Start training...\n");

    }

    train_mlp_mpi(&mlp, dataset.train_set.Y, ETA, EPOCHS, batchSize, &dataset.test_set, myid, procs);

    MPI_Finalize();

    return 0;
}