#ifndef DATA_MANIP_H
#define DATA_MANIP_H
#include <fstream>
#include <algorithm>
#include <random> 
#include <chrono>

/// <summary>
/// Creates dataset array and shuffles data randomly
/// </summary>
/// <param name="sample_array"></param>
/// <param name="sample_size"></param>
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

/// <summary>
/// Generates dataset and splits dataset into training data and testing data
/// </summary>
/// <param name="dataset"></param>
/// <param name="height"></param>
/// <param name="width"></param>
/// <param name="train"></param>
/// <param name="test"></param>
/// <param name="train_size"></param>
void split_data (float * dataset, int height, int width, float * train , float * test, int train_size) {
    // Create dataset array 
    int *sample_array = (int *) malloc(train_size * sizeof(int));
    create_sample_array(sample_array, height);

    // validate shuffle
    // for (int i = 0; i < height; i++) {
    //     printf("%d\n", sample_array[i]);
    // }

    // Split dataset into training and testing data
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

/// <summary>
/// Validates split of dataset into training and testing
/// </summary>
/// <param name="train"></param>
/// <param name="test"></param>
/// <param name="height"></param>
/// <param name="width"></param>
/// <param name="train_size"></param>
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