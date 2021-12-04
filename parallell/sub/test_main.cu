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
    std::string filename = "iris.csv";

    int height = 150, width = 5;
    struct Matrix *dataset = create_matrix(height, width);
    if(!read_data(dataset, filename, false))
    {
        printf("Error reading data\n");
        return -1;
    }
    // print_matrix(dataset);
    // printf("\n");
    // printf("File read successfully\n");
      normalize_data(dataset);

    int train_size = 1;
    
    struct Matrix *train, *test;
    train = create_matrix(train_size, width);
    test = create_matrix(height - train_size, width);
    split_data(dataset, train, test);
    // print_matrix(train);
    // printf("\n");
    // print_matrix(test);
    // float *ground_truth = (float *) malloc(3* train_size * sizeof(float));
    struct Matrix *ground_truth = create_matrix(train_size, 3);
    create_ground_truth(train, ground_truth);
    // print_matrix(ground_truth);
    // // validate_split(train, test, height, width, train_size);
    print_matrix(train);
  
    printf("\n");
    print_matrix(train);
    printf("\n");
    print_matrix(ground_truth);

    int topology[] = {4, 3};
    NeuralNetwork net(topology, 2);
    net.train(train,ground_truth,10000,.001);

    // printf("\n%f\n", dataset[0]);
    // float *inputs, *weights, *outputs;
    // int a = 2, b = 3,c = 4;
    // inputs = (float *) malloc(a * b * sizeof(float));
    // weights = (float *) malloc((b+1) * c * sizeof(float));
    // outputs = (float *) malloc(a * c * sizeof(float));

    // for(int i = 0; i < a * b; i++)
    // {
    //     inputs[i] = i;
    // }
    // for(int i = 0; i < (b+1) * c; i++)
    // {
    //     weights[i] = i;
    // }
    // for(int i = 0; i < a * c; i++)
    // {
    //     outputs[i] = 0;
    // }
    // float *d_inputs, *d_weights, *d_outputs;
    // cudaMalloc(&d_inputs, a * b * sizeof(float));
    // cudaMalloc(&d_weights, (b+1) * c * sizeof(float));
    // cudaMalloc(&d_outputs, a * c * sizeof(float));

    // cudaMemcpy(d_inputs, inputs,  a * b * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_weights, weights,  (b+1) * c * sizeof(float), cudaMemcpyHostToDevice);
    // // printf("\n%f\n", d_weights[0]);
    
    // nn_mat_mul(d_inputs, d_weights, d_outputs, a, b, c);
    // std::cout << "Inputs: " << std::endl;
    // cudaMemcpy(outputs,d_outputs, a * c * sizeof(float), cudaMemcpyDeviceToHost);

    // for (int y = 0; y < a; y++)
    // {
    //     for (int x = 0; x < b; x++)
    //     {
    //         printf("%f ", inputs[y * a + x]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // for (int y = 0; y < b+1; y++)
    // {
    //     for (int x = 0; x < c; x++)
    //     {
    //         printf("%f ", weights[y * (b+1) + x]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // for (int y = 0; y < a; y++) {
    //     for (int x = 0; x < c; x++) {
    //         printf("%f ", outputs[y * c + x]);
    //     }
    //     printf("\n");
    // }
    return 0;
}
