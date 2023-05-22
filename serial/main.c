#include "mlp.h"
#include "../datasets/dataset.h"
#include <omp.h>

int main()
{
    DATASET dataset;
    MLP mlp;

    int hiddenSize = 7;
    float eta = 0.0001;
    int batchSize = 64;
    int epochs = 10;

    prepare_dataset("../datasets/adults/encoded_data_normalized.data", 14, 1, &dataset);

    init_mlp(&mlp, dataset.train_set.X, dataset.train_set.size, dataset.train_set.features, dataset.train_set.outputs, hiddenSize);
    // print_mlp_info(&mlp, 10);

    printf("Start training...\n");
    double start_time = omp_get_wtime();

    train_mlp(&mlp, dataset.train_set.Y, eta, epochs, batchSize);

    double end_time = omp_get_wtime();

    printf("Training complete, time: %.4fs\n", end_time - start_time);

    printf("Start testing...\n");
    test_mlp(&mlp, &dataset.test_set);

    clear_mlp(&mlp, hiddenSize);
    clear_dataset(&dataset);

    return 0;
}