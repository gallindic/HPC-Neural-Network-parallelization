#ifdef __cplusplus

extern "C"
{

#endif

#ifndef MLP_H
#define MLP_H

#include <stdlib.h>
#include "math.h"
#include "dataset.h"

    typedef struct
    {
        int samples;  // Num samples (rows in dataset)
        int features; // Num features
        double **X;   // Matrix features (samples)x(features)
    } InputLayer;

    typedef struct
    {
        int nodes;   // Hidden nodes num
        double **W1; // Hidden layer weights matrix (features)x(hidden nodes num)
        double *b1;  // Hidden layer biases matrix (1)x(hidden nodes num)

    } HiddenLayer;

    typedef struct
    {
        int nodes;   // Output nodes num
        double **W2; // Output layer weights matrix (features)x(output nodes num)
        double *b2;  // Output layer biases matrix (1)x(output nodes num)
    } OutputLayer;

    typedef struct
    {
        InputLayer inputLayer;
        HiddenLayer hiddenLayer;
        OutputLayer outputLayer;
    } MLP;

    void free_2d_double(double **array, int rows)
    {
        for (int i = 0; i < rows; i++)
        {
            free(array[i]);
        }
        free(array);
    }

    void init_mlp(MLP *mlp, double **X, int samples, int features, int outputs, int hiddenSize)
    {
        mlp->inputLayer.features = features;
        mlp->inputLayer.samples = samples;
        mlp->inputLayer.X = (double **)malloc(samples * sizeof(double *));

        for (int i = 0; i < samples; i++)
        {
            mlp->inputLayer.X[i] = (double *)malloc(features * sizeof(double));
            memcpy(&mlp->inputLayer.X[i], &X[i], mlp->inputLayer.features * sizeof(double));
        }

        mlp->hiddenLayer.nodes = hiddenSize;
        mlp->hiddenLayer.W1 = (double **)malloc(features * sizeof(double *));
        mlp->hiddenLayer.b1 = (double *)malloc(hiddenSize * sizeof(double));

        mlp->outputLayer.nodes = outputs;
        mlp->outputLayer.W2 = (double **)malloc(hiddenSize * sizeof(double *));
        mlp->outputLayer.b2 = (double *)malloc(outputs * sizeof(double));

        for (int i = 0; i < features; i++)
        {
            mlp->hiddenLayer.W1[i] = (double *)malloc(hiddenSize * sizeof(double));

            for (int j = 0; j < hiddenSize; j++)
            {
                mlp->hiddenLayer.W1[i][j] = rand_weight();
            }
        }

        for (int i = 0; i < hiddenSize; i++)
        {
            mlp->outputLayer.W2[i] = (double *)malloc(outputs * sizeof(double));

            for (int j = 0; j < outputs; j++)
            {
                mlp->outputLayer.W2[i][j] = rand_weight();
            }

            mlp->hiddenLayer.b1[i] = 0.0;
        }

        for (int i = 0; i < outputs; i++)
        {
            mlp->outputLayer.b2[i] = 0.0;
        }
    }

    void print_mlp_info(MLP *mlp, int truncate)
    {
        printf("\nInput Layer:\n");
        printf("  samples: %d\n", mlp->inputLayer.samples);
        printf("  features: %d\n", mlp->inputLayer.features);
        printf("  X:\n");
        for (int i = 0; i < truncate; i++)
        {
            for (int j = 0; j < mlp->inputLayer.features; j++)
            {
                printf("%f ", mlp->inputLayer.X[i][j]);
            }

            printf("\n");
        }

        printf("\nHidden Layer:\n");
        printf("  nodes: %d\n", mlp->hiddenLayer.nodes);
        printf("  W1:\n");
        for (int i = 0; i < mlp->inputLayer.features; i++)
        {
            for (int j = 0; j < mlp->hiddenLayer.nodes; j++)
            {
                printf("%f ", mlp->hiddenLayer.W1[i][j]);
            }

            printf("\n");
        }
        printf("\n  b1:\n");
        for (int i = 0; i < mlp->hiddenLayer.nodes; i++)
        {
            printf("%f ", mlp->hiddenLayer.b1[i]);
        }
        printf("\n\n");

        printf("Output Layer:\n");
        printf("  nodes: %d\n", mlp->outputLayer.nodes);
        printf("  W2:\n");
        for (int i = 0; i < mlp->hiddenLayer.nodes; i++)
        {
            for (int j = 0; j < mlp->outputLayer.nodes; j++)
            {
                printf("%f ", mlp->outputLayer.W2[i][j]);
            }

            printf("\n");
        }

        printf("\n  b2:\n");
        for (int i = 0; i < mlp->outputLayer.nodes; i++)
        {
            printf("%f ", mlp->outputLayer.b2[i]);
        }

        printf("\n\n");
    }

    void clear_mlp(MLP *mlp, int hiddenSize)
    {
        free_2d_double(mlp->inputLayer.X, mlp->inputLayer.samples);

        free_2d_double(mlp->hiddenLayer.W1, mlp->hiddenLayer.nodes);
        free_2d_double(mlp->outputLayer.W2, mlp->hiddenLayer.nodes);

        free(mlp->hiddenLayer.b1);
        free(mlp->outputLayer.b2);
    }

    void copy_batch_data(double **Xb, double **Yb, double *Y, MLP *mlp, int start, int end)
    {
        int index = 0;

        for (int i = start; i <= end; i++)
        {
            memcpy(Xb[index], mlp->inputLayer.X[i], mlp->inputLayer.features * sizeof(double));
            index++;
        }

        index = 0;

        for (int i = start; i <= end; i++)
        {
            Yb[index][0] = Y[i];
            index++;
        }
    }

    void predict(double **inputs, int n_inputs, double *weights, int n_weights, double *predictions)
    {
        // Calculate predictions for each input
        for (int i = 0; i < n_inputs; i++)
        {
            double y = weights[0]; // Initialize output with bias term
            for (int j = 0; j < n_weights - 1; j++)
            { // Loop over input features
                y += weights[j + 1] * inputs[i][j];
            }
            predictions[i] = (y >= 0) ? 1 : 0; // Apply threshold of 0
        }
    }

    void forward_propagation(double **H, double **Xb, MLP *mlp, double **Y_hat, int batchRows)
    {
        /**
         * Hidden layer
         *
         * H = tanH(XbW1 + b1)
         * Example:
         * Xb = m*n (3x2)
         * W1 = n*p (2x2)
         * H = m*p (2x2)

        printf("Xb = mxn (%dx%d)\n", batchRows, mlp->inputLayer.features);
        printf("W1 = nxp (%dx%d)\n", mlp->inputLayer.features, mlp->hiddenLayer.nodes);
        printf("H = mxp (%dx%d)\n", batchRows, mlp->hiddenLayer.nodes);*/

        calculate_layer_activation(Xb,
                                   mlp->hiddenLayer.W1,
                                   mlp->hiddenLayer.b1,
                                   H,
                                   batchRows,
                                   mlp->inputLayer.features,
                                   mlp->hiddenLayer.nodes);

        /**
         * Output layer
         *
         * Y_hat = tanH(HW2 + b2)
         * Example:
         * H = m*n (2x2)
         * W2 = n*p (2x1)
         * Y_hat = m*p (2x1)

        printf("\n");
        printf("H = mxn (%dx%d)\n", batchRows, mlp->hiddenLayer.nodes);
        printf("W2 = nxp (%dx%d)\n", mlp->hiddenLayer.nodes, mlp->outputLayer.nodes);
        printf("Y_hat = mxp (%dx%d)\n", batchRows, mlp->outputLayer.nodes);*/

        calculate_layer_activation(H,
                                   mlp->outputLayer.W2,
                                   mlp->outputLayer.b2,
                                   Y_hat,
                                   batchRows,
                                   mlp->hiddenLayer.nodes,
                                   mlp->outputLayer.nodes);
    }

    void back_propagation(MLP *mlp, double **Xb, double **H, double *Y, double **Y_hat, double **E, int batchRows, double eta)
    {
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

        for (int i = 0; i < mlp->hiddenLayer.nodes; i++)
        {
            H_transpose[i] = (double *)malloc(batchRows * sizeof(double));
            W2_g[i] = (double *)malloc(mlp->outputLayer.nodes * sizeof(double));
        }

        for (int i = 0; i < mlp->inputLayer.features; i++)
        {
            W1_g[i] = (double *)malloc(mlp->hiddenLayer.nodes * sizeof(double));
            Xb_transpose[i] = (double *)malloc(batchRows * sizeof(double));
        }

        for (int i = 0; i < mlp->outputLayer.nodes; i++)
        {
            W2_transpose[i] = (double *)malloc(mlp->hiddenLayer.nodes * sizeof(double));
        }

        for (int i = 0; i < batchRows; i++)
        {
            Y_hat_squared_minus_one[i] = (double *)malloc(mlp->outputLayer.nodes * sizeof(double));
            H_squared_minus_one[i] = (double *)malloc(mlp->hiddenLayer.nodes * sizeof(double));

            E_Y_hat_hadamard[i] = (double *)malloc(mlp->outputLayer.nodes * sizeof(double));
            He[i] = (double *)malloc(mlp->hiddenLayer.nodes * sizeof(double));
            He_hadamard[i] = (double *)malloc(mlp->hiddenLayer.nodes * sizeof(double));
        }

        calcaulate_error(Y, Y_hat, E, batchRows);

        // Transpose H to get H_transpose values
        transpose(H, H_transpose, batchRows, mlp->hiddenLayer.nodes);

        // Transpose W2 to get W2_transpoe
        transpose(mlp->outputLayer.W2, W2_transpose, mlp->hiddenLayer.nodes, mlp->outputLayer.nodes);

        // Transpose Xb to get Xb_transpose values
        transpose(Xb, Xb_transpose, batchRows, mlp->inputLayer.features);

        // Y_hat squared
        square(Y_hat, batchRows, 1);

        // H squared
        square(H, batchRows, mlp->hiddenLayer.nodes);

        // printf("\n");
        //  (1 - Y_hat squared) & (1 - H squared)
        calculate_one_minus_matrix(Y_hat, Y_hat_squared_minus_one, batchRows, mlp->outputLayer.nodes);
        calculate_one_minus_matrix(H, H_squared_minus_one, batchRows, mlp->hiddenLayer.nodes);

        // Calculate W2_g -> H_transpose(E ◦ (1 − Yˆ^◦2))
        hadamard_product(E, Y_hat_squared_minus_one, E_Y_hat_hadamard, batchRows, mlp->outputLayer.nodes);
        matrix_multiply(H_transpose, E_Y_hat_hadamard, W2_g, mlp->hiddenLayer.nodes, batchRows, mlp->outputLayer.nodes);

        // Calculate b2_g -> 1_transpose(E ◦ (1 − Yˆ^◦2))
        column_sum(E_Y_hat_hadamard, b2_g, batchRows, mlp->outputLayer.nodes);

        // Calculate He -> (E◦(1−Yˆ◦2))W2_transpose
        // He ← He ◦ (1 − H◦2)
        matrix_multiply(E_Y_hat_hadamard, W2_transpose, He, batchRows, mlp->outputLayer.nodes, mlp->hiddenLayer.nodes);
        hadamard_product(He, H_squared_minus_one, He_hadamard, batchRows, mlp->hiddenLayer.nodes);

        // Calculate W1_g ← Xb_transpose He
        matrix_multiply(Xb_transpose, He_hadamard, W1_g, mlp->inputLayer.features, batchRows, mlp->hiddenLayer.nodes);

        // Calculate b1_g -> 1_transpose He
        column_sum(He_hadamard, b1_g, batchRows, mlp->hiddenLayer.nodes);

        // Update weights & bias
        update_weights(mlp->hiddenLayer.W1, W1_g, eta, mlp->inputLayer.features, mlp->hiddenLayer.nodes);
        update_weights(mlp->outputLayer.W2, W2_g, eta, mlp->hiddenLayer.nodes, mlp->outputLayer.nodes);

        update_bias(mlp->hiddenLayer.b1, b1_g, eta, mlp->hiddenLayer.nodes);
        update_bias(mlp->outputLayer.b2, b2_g, eta, mlp->outputLayer.nodes);

        free_2d_double(H_transpose, mlp->hiddenLayer.nodes);
        free_2d_double(W2_transpose, mlp->outputLayer.nodes);
        free_2d_double(Xb_transpose, mlp->hiddenLayer.nodes);
        free_2d_double(W2_g, mlp->hiddenLayer.nodes);
        free_2d_double(W1_g, mlp->inputLayer.features);

        free_2d_double(Y_hat_squared_minus_one, batchRows);
        free_2d_double(He, batchRows);
        free_2d_double(H_squared_minus_one, batchRows);
        free_2d_double(E_Y_hat_hadamard, batchRows);

        free(b1_g);
        free(b2_g);
    }

    void test_mlp(MLP *mlp, Data *test_set)
    {
        double **Y_hat = (double **)malloc(test_set->size * sizeof(double *));
        double **H = (double **)malloc(test_set->size * sizeof(double *));

        for (int i = 0; i < test_set->size; i++)
        {
            Y_hat[i] = (double *)malloc(1 * sizeof(double));
            H[i] = (double *)malloc(mlp->hiddenLayer.nodes * sizeof(double));
        }

        forward_propagation(H, test_set->X, mlp, Y_hat, test_set->size);
        int hits = 0;

        for (int i = 0; i < test_set->size; i++)
        {
            int pred = (Y_hat[i][0] >= 0.5) ? 1 : 0;
            if (pred == (int)test_set->Y[i])
            {
                hits++;
            }
            // printf("Sample %d: True label = %f, Predicted label = %d\n", i, test_set->Y[i], pred);
        }

        printf("Prediction hits: %.3f", (double)hits / test_set->size);
    }

    void train_mlp(MLP *mlp, double *Y, float eta, int epochs, int batchSize)
    {
        int batches = batchSize >= mlp->inputLayer.samples ? 1 : (mlp->inputLayer.samples / batchSize);
        printf("Batches num: %d", batches);
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

                forward_propagation(H, Xb, mlp, Y_hat, batchSize);

                back_propagation(mlp, Xb, H, Y, Y_hat, E, batchSize, eta);
            }
        }

        free_2d_double(Xb, batchSize);
        free_2d_double(Yb, batchSize);
        free_2d_double(H, batchSize);
        free_2d_double(Y_hat, batchSize);
        free_2d_double(E, batchSize);
    }

#endif

#ifdef __cplusplus
}

#endif