#ifndef DATA_MANIP_H
#define DATA_MANIP_H
#include <fstream>
#include <algorithm>
#include <random> 
#include <chrono>

void create_sample_array(int *sample_array, int sample_size)
{
    for(int i = 0; i < sample_size; i++)
    {
        sample_array[i] = i;
    }
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle (sample_array, sample_array + sample_size, std::default_random_engine(seed));
    return;
}

void split_data (float * dataset, int height, int width, float * train , float * test, int train_size) {
    int *sample_array = (int *) malloc(train_size * sizeof(int));
    create_sample_array(sample_array, height);
    // validate shuffle
    // for (int i = 0; i < height; i++) {
    //     printf("%d\n", sample_array[i]);
    // }
    for (int i = 0; i < height; i++) {

        for(int j = 0; j < width; j++) {
            if(i < train_size) {
                train[i * width + j] = dataset[sample_array[i] * width + j];
            } else {
                test[(i - train_size) * width + j] = dataset[sample_array[i] * width + j];
            }
        }
    }
    return;
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



#endif