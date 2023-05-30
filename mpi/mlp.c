#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>
#include "omp.h"
#include "math.h"

#define MASTER 0
#define EPOCHS 5
#define HIDDEN_SIZE 32
#define OUTPUTS 1
#define ETA 0.0001
#define BATCH_SIZE 256
#define DATA_LEN 24421
#define NUM_FEATURES 8

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
    return A_transposed;
}

double ** transpose_mpi(double **A, int rows, int cols, int myid, int procs) {
   double ** finalTransposed;
    if(myid == MASTER){
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

int main(int argc, char* argv[]) {       
	int myid, procs;        
    double **W1;    
    double **W2;    
    double **X;    
    double *Y;    
    double *b1;    
    double *b2;    
    MPI_Status  status;
    
	MPI_Init(&argc, &argv); // initialize MPI 
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);	
	MPI_Comm_size(MPI_COMM_WORLD, &procs);	
    
    char nodename[MPI_MAX_PROCESSOR_NAME];
    int nodename_len;
    MPI_Get_processor_name(nodename, &nodename_len);

    double start_time = MPI_Wtime();

    int batches = (int)(DATA_LEN / BATCH_SIZE);

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        for (int b = 1; b <= batches; b++) {
            X = alloc_2d_double(BATCH_SIZE, NUM_FEATURES);
            Y = (double *)malloc(BATCH_SIZE*sizeof(double));

            if(myid == MASTER){
                W1 = alloc_2d_double(NUM_FEATURES, HIDDEN_SIZE);
                W2 = alloc_2d_double(HIDDEN_SIZE, OUTPUTS);
                b1 = (double *)malloc(HIDDEN_SIZE*sizeof(double));
                b2 = (double *)malloc(OUTPUTS*sizeof(double));
                for(int i = 0; i < NUM_FEATURES; i++){            
                    for(int j = 0; j < HIDDEN_SIZE; j++){
                        W1[i][j] = (double) i;                
                        b1[j] = (double) j;
                    }            
                }
                for(int i = 0; i < BATCH_SIZE; i++){            
                    for(int j = 0; j < NUM_FEATURES; j++){                
                        X[i][j] = (double) i;                
                    }
                }
                for(int i = 0; i < HIDDEN_SIZE; i++){            
                    for(int j = 0; j < OUTPUTS; j++){                
                        W2[i][j] = (double) i;      
                        b2[j] = (double) j;          
                    }
                }
                for (int i = 0; i < BATCH_SIZE; i++)
                    Y[i] = (double) (i % 2);
            }


            //** Forward propagation
            // H <- tanh(X_b * W_1 + b_1)
            double **H;
            H = calculate_layer_activation_mpi(W1, X, b1, H, BATCH_SIZE, NUM_FEATURES, HIDDEN_SIZE, myid, procs);

            //** Back propagation
            // E
            double **E;
            E = calculate_error_mpi(H, H, BATCH_SIZE, NUM_FEATURES, myid, procs);
            // Transpose H
            double **H_transposed;
            H_transposed = transpose_mpi(H, BATCH_SIZE, NUM_FEATURES, myid, procs);
            // Transpose W1
            double **W1_transposed;
            W1_transposed = transpose_mpi(W1, BATCH_SIZE, NUM_FEATURES, myid, procs);
            // Transpose Xb
            double **X_transposed;
            X_transposed = transpose_mpi(X, BATCH_SIZE, NUM_FEATURES, myid, procs);
            // Square H
            double **H_squared;
            H_squared = square_mpi(H, BATCH_SIZE, NUM_FEATURES, myid, procs);
            // Square X
            double **X_squared;
            X_squared = square_mpi(X, BATCH_SIZE, NUM_FEATURES, myid, procs);
            // One minus H
            double **one_minus_H;
            one_minus_H = calculate_one_minus_matrix_mpi(H, BATCH_SIZE, NUM_FEATURES, myid, procs);
            // One minus X
            double **one_minus_X;
            one_minus_X = calculate_one_minus_matrix_mpi(X, BATCH_SIZE, NUM_FEATURES, myid, procs);
            // Haddamard 1
            double **He;
            He = haddamard_product_mpi(X, H, BATCH_SIZE, NUM_FEATURES, myid, procs);
            H = matmul_mpi(W1, X, BATCH_SIZE, NUM_FEATURES, HIDDEN_SIZE, myid, procs);
            // Haddamard 2
            H = matmul_mpi(W1, X, BATCH_SIZE, NUM_FEATURES, HIDDEN_SIZE, myid, procs);
            He = haddamard_product_mpi(X, H, BATCH_SIZE, NUM_FEATURES, myid, procs);
            // Matrix multiplication
            H = matmul_mpi(W1, X, BATCH_SIZE, NUM_FEATURES, HIDDEN_SIZE, myid, procs);
        }        
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(myid == MASTER){
        double end_time = MPI_Wtime();
        printf("time = %f\n", end_time-start_time);
    }

	MPI_Finalize();
    
	return 0;
}