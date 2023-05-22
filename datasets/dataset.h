#ifdef __cplusplus

extern "C"
{

#endif

#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define MAX_LINE 1024
#define SPLIT_RATIO 0.75

    typedef struct
    {
        int size;
        int features;
        int outputs;
        double **X;
        double *Y;
    } Data;

    typedef struct
    {
        Data train_set;
        Data test_set;
    } DATASET;

    int get_num_lines(FILE *file, char *line)
    {
        int num_lines = 0;

        while (fgets(line, 1024, file) != NULL)
        {
            num_lines++;
        }

        fseek(file, 0, SEEK_SET);

        return num_lines;
    }

    void read_dataset(FILE *file, char *line, int num_features, DATASET *dataset)
    {
        const char *delimiter = " ";
        int curr_line = 1;
        int train_set_idx = 0, test_set_idx = 0;

        while (fgets(line, 1024, file) != NULL)
        {
            char *token = strtok(line, delimiter);
            char *features[num_features];
            char *output;

            int i = 0;

            while (token != NULL)
            {
                if (i == num_features)
                {
                    output = token;
                }
                else
                {
                    features[i] = token;
                }

                token = strtok(NULL, delimiter);
                i++;
            }

            for (int j = 0; j < num_features; j++)
            {
                if (curr_line <= dataset->train_set.size)
                {
                    dataset->train_set.X[train_set_idx][j] = strtod(features[j], NULL);
                }
                else
                {
                    dataset->test_set.X[test_set_idx][j] = strtod(features[j], NULL);
                }
            }

            if (curr_line <= dataset->train_set.size)
            {
                dataset->train_set.Y[train_set_idx] = strtod(output, NULL);
                train_set_idx++;
            }
            else
            {
                dataset->test_set.Y[test_set_idx] = strtod(output, NULL);
                test_set_idx++;
            }

            curr_line++;
        }

        fclose(file);
    }

    void init_set(Data *set, int num_features)
    {
        set->X = (double **)malloc(set->size * sizeof(double *));

        for (int i = 0; i < set->size; i++)
        {
            set->X[i] = (double *)malloc(num_features * sizeof(double));
        }

        set->Y = (double *)malloc(set->size * sizeof(double));
    }

    void prepare_dataset(const char *filename, int num_features, int outputs, DATASET *dataset)
    {
        FILE *file = fopen(filename, "r");

        if (file == NULL)
        {
            printf("Error opening file %s\n", filename);
            return;
        }

        printf("Reading dataset file %s\n", filename);

        char line[MAX_LINE];
        int num_lines = get_num_lines(file, line);

        printf("Number of lines: %d\n", num_lines);
        printf("Splitting dataset into train/test\n");

        int train_size = ceil(num_lines * SPLIT_RATIO);
        int test_size = num_lines - train_size;

        printf("Train set size %d\nTest set size %d\n", train_size, test_size);

        dataset->train_set.size = train_size;
        dataset->train_set.features = num_features;
        dataset->train_set.outputs = outputs;

        dataset->test_set.size = test_size;
        dataset->test_set.features = num_features;
        dataset->test_set.outputs = outputs;

        init_set(&dataset->train_set, num_features);
        init_set(&dataset->test_set, num_features);

        read_dataset(file, line, num_features, dataset);

        printf("Dataset prepared\n");
    }

    void clear_dataset(DATASET *dataset)
    {
        free(dataset->train_set.X);
        free(dataset->train_set.Y);

        free(dataset->test_set.X);
        free(dataset->test_set.Y);
    }

#endif

#ifdef __cplusplus
}

#endif
