#include "mlp.h"
#include "dataset.h"

int main()
{
    DATASET dataset;
    MLP mlp;

    int hiddenSize = 3;
    float eta = 0.00000001;
    int batchSize = 64;
    int epochs = 2;

    prepare_dataset("adults/encoded_data_normalized.data", 14, 1, &dataset);

    init_mlp(&mlp, dataset.train_set.X, dataset.train_set.size, dataset.train_set.features, dataset.train_set.outputs, hiddenSize);
    // print_mlp_info(&mlp, 10);

    printf("Start training...\n");
    train_mlp(&mlp, dataset.train_set.Y, eta, epochs, batchSize);
    printf("Training complete...\n");

    printf("Start testing...\n");
    test_mlp(&mlp, &dataset.test_set);

    clear_mlp(&mlp, hiddenSize);
    clear_dataset(&dataset);

    return 0;
}