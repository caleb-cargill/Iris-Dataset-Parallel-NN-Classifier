#include <fstream>
#include "file_io.h"
#include "data_manip.h"
#include "neural_net.h"
#include "mat_mult.h"
#include "cuda.h"
#include "matrix.h"
#include <iostream>

int main(void)
{   
    int num_samples = 4;
    int width = 3;
    struct Matrix *output, *ground_truth, *error;
    output = create_matrix(num_samples, width);
    ground_truth = create_matrix(num_samples, width);
    error = create_matrix(num_samples, width);

    struct Matrix *d_output, *d_ground_truth, *d_error;
    d_output = create_cuda_matrix(num_samples, width);
    d_ground_truth = create_cuda_matrix(num_samples, width);
    d_error = create_cuda_matrix(num_samples, width);

    // float *output, *ground_truth, *error;
    // float *d_output, *d_ground_truth, *d_error;

    // output = (float*)malloc(sizeof(float)*num_samples*width);
    // ground_truth = (float*)malloc(sizeof(float)*num_samples*width);
    // error = (float*)malloc(sizeof(float)*num_samples*width);

    // cudaMalloc(&d_output, sizeof(float)*num_samples*width);
    // cudaMalloc(&d_ground_truth, sizeof(float)*num_samples*width);
    // cudaMalloc(&d_error, sizeof(float)*num_samples*width);

    for(int i = 0; i < num_samples; i++)
    {
        for(int j = 0; j < width; j++)
        {
            output->data[i*width + j] = i*width + j + i*2;
            ground_truth->data[i*width + j] = i*width + j;
        }
    }
    print_matrix(output);
    printf("\n");
    print_matrix(ground_truth);

    struct Matrix * avg = create_cuda_matrix(1, width);

    int num_threads = output->width;
    int num_blocks = 1;
    if (num_threads > 512)
    {
        num_threads = 512;
        num_blocks = ceil(float(output->width) / 512.0);
    }
    print_matrix(output);
    d_output = send_matrix_to_gpu(output);
    average_columns_kernal<<<num_blocks, num_threads>>>(d_output->data, avg->data, output->height,output->width);
    print_matrix(avg);
    // element_wise_multiplication(send_matrix_to_gpu(output), send_matrix_to_gpu(ground_truth), d_error);
    // print_matrix(d_error);
    
    // print_matrix(output);
    // for (int i = 0; i < num_samples; i++)
    // {
    //     for (int j = 0; j < width; j++)
    //     {
    //         printf("%f ", output[i*width + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // for (int i = 0; i < num_samples; i++)
    // {
    //     for (int j = 0; j < width; j++)
    //     {
    //         printf("%f ", ground_truth[i*width + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // cudaMemcpy(d_output, output, sizeof(float)*num_samples*width, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_ground_truth, ground_truth, sizeof(float)*num_samples*width, cudaMemcpyHostToDevice);

    // int num_threads = num_samples;
    // int num_blocks = 1;
    // if (num_samples > 512)
    // {
    //     num_threads = 512;
    //     num_blocks = ceil(float(num_samples) / 512.0);
    // }
    // calculate_error_kernal<<<num_blocks, num_threads>>>(d_output, d_ground_truth, d_error, width, num_samples);
    // cudaMemcpy(error, d_error, sizeof(float)*num_samples*width, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < num_samples; i++)
    // {
    //     for (int j = 0; j < width; j++)
    //     {
    //         printf("%f ", error[i*width + j]);
    //     }
    //     printf("\n");
    // }

}