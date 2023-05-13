#ifdef __cplusplus

extern "C"
{

#endif

#ifndef MATH_H
#define MATH_H

#include <math.h>
#include <stdio.h>

    double tanh_activation(double x)
    {
        return tanh(x);
    }

    double rand_weight()
    {
        return 2.0 * (double)rand() / (double)RAND_MAX - 1.0;
    }

    void matrix_multiply(double **A, double **B, double **C, int m, int n, int p)
    {
        int i, j, k;

        for (i = 0; i < m; i++)
        {
            for (j = 0; j < p; j++)
            {
                C[i][j] = 0;
                for (k = 0; k < n; k++)
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    void calculate_layer_activation(double **A, double **weights, double *bias, double **C, int m, int n, int p)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < n; k++)
                {
                    sum += A[i][k] * weights[k][j];
                }

                C[i][j] = tanh_activation(sum + bias[j]);
            }
        }
    }

    void calcaulate_error(double *Y, double **Y_hat, double **E, int m)
    {
        for (int i = 0; i < m; i++)
        {
            E[i][0] = Y[i] - Y_hat[i][0];
        }
    }

    void transpose(double **A, double **B, int rows, int cols)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                B[j][i] = A[i][j];
            }
        }
    }

    void square(double **A, int rows, int cols)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                A[i][j] = A[i][j] * A[i][j];
            }
        }
    }

    void calculate_one_minus_matrix(double **A, double **B, int rows, int cols)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                B[i][j] = 1 - A[i][j];
            }
        }
    }

    void hadamard_product(double **A, double **B, double **C, int rows, int cols)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                C[i][j] = A[i][j] * B[i][j];
            }
        }
    }

    void column_sum(double **A, double *sums, int rows, int cols)
    {
        for (int j = 0; j < cols; j++)
        {
            double col_sum = 0.0;

            for (int i = 0; i < rows; i++)
            {
                col_sum += A[i][j];
            }

            sums[j] = col_sum;
        }
    }

    void update_weights(double **W, double **Wg, float eta, int rows, int cols)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                W[i][j] -= eta * Wg[i][j];
            }
        }
    }

    void update_bias(double *b, double *bg, float eta, int size)
    {
        for (int i = 0; i < size; i++)
        {
            b[i] -= eta * bg[i];
        }
    }

    int tanh_to_binary(double output)
    {
        double threshold = 0.5;
        int binary_output = (output >= threshold) ? 1 : 0;
        return binary_output;
    }

#endif

#ifdef __cplusplus
}

#endif