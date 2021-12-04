#ifndef MAT_MULT_H
#define MAT_MULT_H
#include <iostream>
#include "cuda.h"
#include "matrix.h"

__global__ void matrix_multiplication_kernel(float* A, float* B, float* C, int a, int b, int c) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < a && COL < c) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < b; i++) {
            tmpSum += A[ROW * b + i] * B[i * c + COL];
        }
    }
    C[ROW * c + COL] = tmpSum;
}

void matrix_multiplication(float* A, float* B, float* C, int a, int b, int c) {
    // matrix A size = a x b
    // matrix B size = b x c
    // matrix C size = a x c
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, a * b * sizeof(float));
    cudaMalloc(&d_b, b * c * sizeof(float));
    cudaMalloc(&d_c, a * c * sizeof(float));

    cudaMemcpy(d_a, A,  a * b * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B,  b * c * sizeof(float), cudaMemcpyHostToDevice);

    // for(int y = 0; y < a; y++) {
    //     for(int x = 0; x < b; x++){
    //         printf("%f ", A[y * b + x]);
    //     }
    //     printf("\n");
    // }

    dim3 threadsPerBlock(c, a);
    dim3 blocksPerGrid(1, 1);
        if (c*a > 512){
            threadsPerBlock.x = 512;
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(double(c)/double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(a)/double(threadsPerBlock.y));
        }
    matrix_multiplication_kernel<<<blocksPerGrid,threadsPerBlock>>>(d_a, d_b, d_c, a, b, c);
    cudaMemcpy(C, d_c, a * c * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

__global__ void nn_matrix_multiplication_kernel(float* inputs, float* weights, float* outputs, int a, int b, int c) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < a && COL < c) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < b; i++) {
            tmpSum += inputs[ROW * b + i] * weights[i * c + COL];
        }
        tmpSum += 1.0 * weights[b * c + COL];
    }
    // sigmoid activation function
    tmpSum = 1.0 / (1.0 + expf(-tmpSum));
    
    outputs[ROW * c + COL] = tmpSum;
}

void nn_mat_mul(struct Matrix* inputs, struct Matrix* weights, struct Matrix* outputs) {
    // printf("%d %d %d\n", inputs->height, inputs->width, outputs->height);
    dim3 threadsPerBlock(outputs->width, inputs->height);
    // std::cout << "threadsPerBlock: " << threadsPerBlock.x << " " << threadsPerBlock.y << std::endl;
    dim3 blocksPerGrid(1, 1);
        if (outputs->width * inputs->height > 512){
            threadsPerBlock.x = 512;
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(outputs->width/double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(inputs->height)/double(threadsPerBlock.y));
        }
    nn_matrix_multiplication_kernel<<<blocksPerGrid,threadsPerBlock>>>(inputs->data, weights->data, outputs->data, inputs->height, inputs->width, outputs->width);
}

__global__ void element_wise_multiplication_kernal(float* A, float* B, float* C, int width, int height) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < height && COL < width) {
        C[ROW * width + COL] = A[ROW * width + COL] * B[ROW * width + COL];
    }
}

__global__ void element_wise_multiplication_add1_kernal(float* A, float* B, float* C, int width, int height) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < height && COL < width) {
        if (COL == width - 1) {
            C[ROW * width + COL] = A[ROW * width + COL];
        } else {
            C[ROW * width + COL] = A[ROW * width + COL] * B[ROW * width + COL];
        }
        
    }
}

__global__ void calculate_output_error_kernal(float* output, float *ground_truth, float *error, int width,int total)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < total)
    {
        for (int i = 0; i < width; i ++)
        {
            float temp = (output[idx*width+i] - ground_truth[idx*width+i]);
            // temp = temp * temp;
            error[idx*width+i] = temp;
            error[idx*width+i] *= (output[idx*width+i])*(1-output[idx*width+i]);
        }
    }
}

__global__ void calculate_hidden_error_kernal(float* error, float *weights, float *next_error, float * outputs, int current_width, int next_width,int total)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < total)
    {
        for (int i = 0; i < current_width; i++)
        {   
            float acc = 0;
            for (int j = 0; j < next_width; j++)
            {
                acc += weights[i*next_width+j]*next_error[idx*next_width+j];
            }
            error[idx*current_width+i] = acc;
            if (i != current_width - 1)
            {
                error[idx*current_width+i] *= outputs[idx*current_width+i]*(1-outputs[idx*current_width+i]);

            }
        }
    }


void element_wise_multiplication(struct Matrix* A, struct Matrix* B, struct Matrix* C, bool add1) {
    if (add1)
    {
        dim3 threadsPerBlock(A->width, A->height);
        dim3 blocksPerGrid(1, 1);
        if (A->width * A->height > 512){
            threadsPerBlock.x = 512;
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(double(A->width)/double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(A->height)/double(threadsPerBlock.y));
        }
        element_wise_multiplication_add1_kernal<<<blocksPerGrid,threadsPerBlock>>>(A->data, B->data, C->data, A->width, A->height);
    }
    else
    {
        if (A->width != B->width || A->height != B->height || A->width != C->width || A->height != C->height) {
        printf("Error: matrices are not the same size\n");
        exit(1);
    }

        dim3 threadsPerBlock(A->width, A->height);
        dim3 blocksPerGrid(1, 1);
        if (A->width * A->height > 512){
            threadsPerBlock.x = 512;
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(double(A->width)/double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(A->height)/double(threadsPerBlock.y));
        }
        element_wise_multiplication_kernal<<<blocksPerGrid,threadsPerBlock>>>(A->data, B->data, C->data, A->width, A->height);
    }
}

__global__ void update_weights_kernal(float* weights, float* delta, int height, int width, float learning_rate) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < height && COL < width) {
        weights[ROW * width + COL] -= (learning_rate *  delta[ROW]);
    }
    
}

__global__ void average_columns_kernal(float* matrix, float * averages, int height, int width)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < width)
    {
        float acc = 0;
        for (int i = 0; i < height; i++)
        {
            acc += matrix[width*i + idx];
        }
        averages[idx] = acc/float(height);
    }
}

#endif