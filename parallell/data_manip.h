#ifndef DATA_MANIP_H
#define DATA_MANIP_H
#include <fstream>
#include <algorithm>
#include <random> 
#include <chrono>
#include "matrix.h"

void create_sample_array(int *sample_array, int sample_size)
{
    for(int i = 0; i < sample_size; i++)
    {
        sample_array[i] = i;
    }
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle (sample_array, sample_array + sample_size, std::default_random_engine(seed));

}

void split_data (struct Matrix * dataset, struct Matrix *train, struct Matrix *test) {
    int *sample_array = (int *) malloc(dataset->height * sizeof(int));
    create_sample_array(sample_array, dataset->height);
    for (int i = 0; i < dataset->height; i++) {

        for(int j = 0; j < dataset->width; j++) {
            if(i < train->height) {
                train->data[i * dataset->width + j] = dataset->data[sample_array[i] * dataset->width + j];
            } else {
                test->data[(i - train->height) * dataset->width + j] = dataset->data[sample_array[i] * dataset->width + j];
            }
        }
    }
}

void validate_split(float *train, float *test, int height, int width, int train_size)
{
    printf("Train\n");
    for (int i = 0; i < height; i++) {
        if (i == train_size) {
            printf("Test\n");
        }
        for(int j = 0; j < width; j++) {
            if(i < train_size) {
                printf("%f, ", train[i * width + j]);
            } else {
                printf("%f, ", test[(i - train_size) * width + j]);
            }
        }
        printf("\n");
    }
}

void normalize_data(struct Matrix *dataset)
{
    struct Matrix *max_val = create_matrix(1, dataset->width);
    struct Matrix *min_val = create_matrix(1, dataset->width);
    for (int i = 0; i < dataset->width; i++) {
        max_val->data[i] = dataset->data[i];
        min_val->data[i] = dataset->data[i];
    }
    for (int i = 0; i < dataset->height; i++) {
        for (int j = 0; j < dataset->width - 1; j++) {
            if (dataset->data[i * dataset->width + j] < min_val->data[j]) {
                min_val->data[j] = dataset->data[i * dataset->width + j];
            }
            if (dataset->data[i * dataset->width + j] > max_val->data[j]) {
                max_val->data[j] = dataset->data[i * dataset->width + j];
            }
        }
    }

    for (int i = 0; i < dataset->height; i++) {
        for (int j = 0; j < dataset->width - 1; j++) {
            dataset->data[i * dataset->width + j] = (dataset->data[i * dataset->width + j] - min_val->data[j]) / (max_val->data[j] - min_val->data[j]);
        }
    }
}


void create_ground_truth(struct Matrix * input, struct Matrix * output)
{
    for (int x = 0; x < input->height; x++) {
            if (input->data[x * input->width + 4] - 0 < .01) {
                output->data[x * 3 + 0] = 1;
                output->data[x * 3 + 1] = 0;
                output->data[x * 3 + 2] = 0;
            }
            else if (input->data[x * input->width + 4] - 1 < .01) {
                output->data[x * 3 + 0] = 0;
                output->data[x * 3 + 1] = 1;
                output->data[x * 3 + 2] = 0;
            }
            else if (input->data[x * input->width + 4] - 2 < .01) {
                output->data[x * 3 + 0] = 0;
                output->data[x * 3 + 1] = 0;
                output->data[x * 3 + 2] = 1;
            }
            else {
                output->data[x * 3 + 0] = 0;
                output->data[x * 3 + 1] = 0;
                output->data[x * 3 + 2] = 0;
            }
            
    }

}

#endif