#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "../datasets/dataset.h"
#include "../serial/mlp.h"

#define EPOCHS 1
#define HIDDEN_SIZE 7
#define OUTPUTS 1
#define ETA 0.0001
#define NUM_FEATURES 14

int main(int argc, char** argv) {
    DATASET dataset;
    MLP mlp;

    int rank, numProcesses; // Process rank, Number of processes

    MPI_Init(&argc, &argv); // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses); // Get number of processes

    int batchSize = numProcesses;
    float aa[dataset.train_set.size/numProcesses];

    // Prepare dataset in the root process
    if (rank == 0) 
        prepare_dataset("../datasets/adults/encoded_data.data", NUM_FEATURES, OUTPUTS, &dataset);    
    
    MPI_Barrier(MPI_COMM_WORLD);

    int batches = batchSize >= dataset.train_set.size ? 1 : ((dataset.train_set.size + batchSize - 1) / batchSize);

    // Scatter rows from the root process to all other processes
    int rowsPerProcess = dataset.train_set.size / numProcesses;
    int remainingRows = dataset.train_set.size % numProcesses;

    MPI_Scatter(dataset.train_set.X, rowsPerProcess, MPI_FLOAT,
                aa, rowsPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
     
    // Divide the dataset into batches in the root process
    if (rank == 0)
        printf("Batches: %d; Batch size: %d\n", batches, batchSize);

    // Access batch data here
    for (int batch = 0; batch < 1; batch++) {
        // Gather the batch data from all processes
        MPI_Gather(aa, rowsPerProcess, MPI_FLOAT, NULL, rowsPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);

        // Process one row at a time in each process
        for (int row = 0; row < rowsPerProcess; row++) {
            // Each process processes one row from the batch
            float row_data = aa[row];
            printf("Rank %d - Processing Row: %f\n", rank, row_data);
            // Perform computations on row_data
        }

        // Continue with other computations or communication
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize(); // Finalize MPI

    return 0;
}