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

    for(int y = 0; y < a; y++) {
        for(int x = 0; x < b; x++){
            printf("%f ", A[y * b + x]);
        }
        printf("\n");
    }

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
    printf("%d %d %d\n", inputs->height, inputs->width, outputs->height);
    dim3 threadsPerBlock(outputs->width, inputs->height);
    std::cout << "threadsPerBlock: " << threadsPerBlock.x << " " << threadsPerBlock.y << std::endl;
    dim3 blocksPerGrid(1, 1);
        if (outputs->width * inputs->height > 512){
            threadsPerBlock.x = 512;
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(outputs->width/double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(inputs->height)/double(threadsPerBlock.y));
        }
    nn_matrix_multiplication_kernel<<<blocksPerGrid,threadsPerBlock>>>(inputs->data, weights->data, outputs->data, inputs->height, inputs->width, outputs->width);
}

#endif