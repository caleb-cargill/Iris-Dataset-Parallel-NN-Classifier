#ifndef ARRAY_H
#define ARRAY_H

#include "cuda.h"

struct Matrix {
    int width;
    int height;
    bool is_GPU;
    float *data;
};

// struct Matrix *create_matrix(int height, int width);
// struct Matrix *create_cuda_matrix(int height, int width);
// void matrix_destroy(Matrix *matrix);
// void print_matrix(Matrix *matrix);
// void send_matrix_to_gpu(Matrix *matrix, Matrix *d_matrix);
// struct Matrix *send_matrix_to_gpu(Matrix *matrix);
// void retrieve_matrix_from_gpu(Matrix *matrix, Matrix *d_matrix);
// Matrix *retrieve_matrix_from_gpu(Matrix *d_matrix);


struct Matrix* create_matrix(int height, int width) {
    Matrix *matrix = (Matrix *) malloc(sizeof(Matrix));
    matrix->width = width;
    matrix->height = height;
    matrix->is_GPU = false;
    matrix->data = (float *) malloc(sizeof(float) * width * height);
    return matrix;
}

struct Matrix* create_cuda_matrix(int height, int width) {
    Matrix *matrix = (Matrix *) malloc(sizeof(Matrix));
    matrix->width = width;
    matrix->height = height;
    matrix->is_GPU = true;
    cudaMalloc(&matrix->data, width * height * sizeof(float));
    return matrix;
}

void send_matrix_to_gpu(Matrix *matrix, Matrix *d_matrix) {
    if (matrix->width != d_matrix->width || matrix->height != d_matrix->height) {
        printf("Matrix dimensions do not match!\nCPU: %d x %d\nGPU: %d x %d\n", matrix->height, matrix->width, d_matrix->height, d_matrix->width);
        exit(1);
    }
    cudaMemcpy(d_matrix->data, matrix->data, matrix->width * matrix->height * sizeof(float), cudaMemcpyHostToDevice);
}

struct Matrix* send_matrix_to_gpu(Matrix *matrix) {
    Matrix *d_matrix = (Matrix *) malloc(sizeof(Matrix));
    d_matrix->width = matrix->width;
    d_matrix->height = matrix->height;
    d_matrix->is_GPU = true;
    cudaMalloc(&d_matrix->data, matrix->width * matrix->height * sizeof(float));
    cudaMemcpy(d_matrix->data, matrix->data, matrix->width * matrix->height * sizeof(float), cudaMemcpyHostToDevice);
    return d_matrix;
}

void retrieve_matrix_from_gpu(Matrix *matrix, Matrix *d_matrix) {
    if (matrix->width != d_matrix->width || matrix->height != d_matrix->height) {
        printf("Matrix dimensions do not match!\nCPU: %d x %d\nGPU: %d x %d\n", matrix->height, matrix->width, d_matrix->height, d_matrix->width);
        exit(1);
    }
    cudaMemcpy(matrix->data, d_matrix->data, matrix->width * matrix->height * sizeof(float), cudaMemcpyDeviceToHost);
}

struct Matrix* retrieve_matrix_from_gpu(Matrix *d_matrix) {
    Matrix *matrix = (Matrix *) malloc(sizeof(Matrix));
    matrix->width = d_matrix->width;
    matrix->height = d_matrix->height;
    matrix->is_GPU = false;
    matrix->data = (float *) malloc(sizeof(float) * matrix->width * matrix->height);
    cudaMemcpy(matrix->data, d_matrix->data, matrix->width * matrix->height * sizeof(float), cudaMemcpyDeviceToHost);
    return matrix;
}

void matrix_destroy(Matrix *matrix) {
    if (matrix->is_GPU) {
        cudaFree(matrix->data);
    } else {
        free(matrix->data);
    }
    free(matrix);
}

void print_matrix(Matrix *matrix) {
    if (matrix->is_GPU) {
        printf("GPU matrix:\n");
        Matrix *cpu_matrix = retrieve_matrix_from_gpu(matrix);
        print_matrix(cpu_matrix);
        matrix_destroy(cpu_matrix);
        return;
    }
    else{
        for (int i = 0; i < matrix->height; i++) {
            for (int j = 0; j < matrix->width; j++) {
                printf("%f ", matrix->data[i * matrix->width + j]);
            }
            printf("\n");
        }
    }
    
}

void print_matrix_dims(Matrix *matrix) {
    printf("Matrix dimensions: %d x %d\n", matrix->height, matrix->width);
}

#endif