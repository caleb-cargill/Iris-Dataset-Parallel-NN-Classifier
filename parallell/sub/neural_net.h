#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <random> 
#include <algorithm>
#include "mat_mult.h"
#include "matrix.h"
#include "cuda.h"

__global__ void calculate_output_error_kernal(float* output, float *ground_truth, float *error, int width,int total)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < total)
    {
        for (int i = 0; i < width; i ++)
        {
            float temp = (output[idx*width+i] - ground_truth[idx*width+i]);
            // temp = temp * temp;
            error[idx*width+i] = temp*(1-temp);
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
            if (i != current_width-1)
            {
                error[idx*current_width+i] *= outputs[idx*current_width+i]*(1-outputs[idx*current_width+i]);
            }

        }
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
        averages[idx] = acc/width;
    }
}


class NeuralNetwork {
    private:
        int *topology;
        int num_inputs, num_outputs;
        int num_hidden_layers, num_layers;
        struct Matrix *error, *d_error;
        struct Matrix **weights, **layer_outputs, **layer_errors;
    public:
        NeuralNetwork(int *topology, int num_layers)
        {
            printf("Constructing Neural Network\n");
            this->num_layers = num_layers;
            this->topology = topology;
            this->num_inputs = topology[0];
            this->num_outputs = topology[num_layers - 1];
            this->num_hidden_layers = num_layers - 2;

            this->weights = new struct Matrix*[num_layers - 1];
            this->layer_errors = new struct Matrix*[num_layers - 1];
            this->layer_outputs = new struct Matrix*[num_layers - 1];
            std::default_random_engine generator;
            // create weights matrix for each layer
            for (int i = 0; i < num_layers - 1; i++)
            {
                // weights range from -1/sqrt(n) to 1/sqrt(n) where n is the number of nodes in the previous layer
                std::uniform_real_distribution<float> distribution(-1.0/sqrt(topology[i]),1.0/sqrt(topology[i]));

                printf("num weights %i\n",(topology[i] + 1) * topology[i + 1]);
                struct Matrix * temp = create_matrix(topology[i] + 1, topology[i + 1]);

                for(int y = 0; y < temp->height* temp->width; y++)
                {
                    temp->data[y] = distribution(generator);
                }
                this->weights[i] = send_matrix_to_gpu(temp);
                print_matrix(temp);
                printf("\n");
                // to check if weigths are being set correctly
                // float *temp2 = (float *) malloc((topology[i] + 1) * topology[i + 1] * sizeof(float));
                // cudaMemcpy(temp2,this->weights[i], (topology[i] + 1) * topology[i + 1] * sizeof(float), cudaMemcpyDeviceToHost);
                // for(int y = 0; y < (topology[i] + 1) * topology[i + 1]; y++)
                // {
                //     printf("%f\n",temp2[y]);
                // }
            }

        }
        NeuralNetwork::~NeuralNetwork()
        {
            printf("Destroying Neural Network\n");
            for (int i = 0; i < num_layers - 1; i++)
            {
                cudaFree(this->weights[i]);
            }
            delete[] this->weights;
            delete[] this->topology;
        }

        void forward_propagate(struct Matrix *input)
        {
            int i = 0;
            printf("i: %i\n",i);
            nn_mat_mul(input, this->weights[i], this->layer_outputs[i]);
            i++;
            printf("i: %i\n",i);
            for (; i < this->num_layers - 1; i++)
            {
                nn_mat_mul(this->layer_outputs[i-1], this->weights[i], this->layer_outputs[i]);
                printf("i: %i\n",i);
                print_matrix(this->layer_outputs[i]);
            }
        }

        void calculate_output_error(struct Matrix *ground_truth)
        {
            this->d_error = create_cuda_matrix(ground_truth->height, ground_truth->width);

            int num_threads = ground_truth->height;
            int num_blocks = 1;
            if (ground_truth->height > 512)
            {
                num_threads = 512;
                num_blocks = ceil(float(ground_truth->height) / 512.0);
            }
            
            // print_matrix(ground_truth);
            // printf("\n");
            // print_matrix(this->layer_outputs[this->num_layers - 2]);
            // printf("\n");
            // print_matrix(this->d_error);

            calculate_output_error_kernal<<<num_blocks, num_threads>>>(this->layer_outputs[this->num_layers - 2]->data, ground_truth->data, this->d_error->data,this->num_outputs,ground_truth->height);
            this->error = retrieve_matrix_from_gpu(this->d_error);
        }

        void calculate_hidden_errors()
        {
            int num_samples =  this->layer_errors[0]->height;
            for (int x = this->num_layers - 2; x >= 0; x--)
            {
                printf("x: %i\n",x);
                int num_threads = num_samples;
                int num_blocks = 1;
                if (num_samples > 512)
                {
                    num_threads = 512;
                    num_blocks = ceil(float(num_samples) / 512.0);
                }
                if (x == this->num_layers - 2)
                {
                    // print_matrix_dims(this->layer_errors[x]);
                    // print_matrix(this->layer_errors[x]);
                    // printf("\n");
                    // print_matrix_dims(this->weights[x]);
                    // print_matrix(this->weights[x]);
                    // printf("\n");
                    // print_matrix_dims(this->d_error);
                    // print_matrix(this->d_error);
                    // printf("\n");
                    // print_matrix_dims(this->layer_outputs[x]);
                    // print_matrix(this->layer_outputs[x]);
                    // printf("\n");
                    calculate_hidden_error_kernal<<<num_blocks, num_threads>>>(this->layer_errors[x]->data, this->weights[x]->data, this->d_error->data, this->layer_outputs[x]->data ,this->topology[x] + 1, this->topology[x + 1], num_samples);
                }
                else
                {
                    // print_matrix_dims(this->layer_errors[x]);
                    // print_matrix(this->layer_errors[x]);
                    // printf("\n");
                    // print_matrix_dims(this->weights[x]);
                    // print_matrix(this->weights[x]);
                    // printf("\n");
                    // print_matrix_dims(this->d_error);
                    // print_matrix(this->d_error);
                    // printf("\n");
                    calculate_hidden_error_kernal<<<num_blocks, num_threads>>>(this->layer_errors[x]->data, this->weights[x]->data, this->layer_errors[x+1]->data, this->layer_outputs[x]->data ,this->topology[x] + 1, this->topology[x + 1], num_samples);
                }
                print_matrix(this->layer_errors[x]);
                // printf("\n\n\n");
            }

        }

        void adjust_weights(float learning_rate)
        {
            for (int i = 0; i < this->num_layers - 1; i++)
            {   
                struct Matrix *averages = create_cuda_matrix(1, this->topology[i] + 1);
                int num_threads = this->topology[i] + 1;
                int num_blocks = 1;
                if (num_threads > 512)
                {
                    num_threads = 512;
                    num_blocks = ceil(float(this->topology[i] + 1;) / 512.0);
                }
                average_columns_kernal<<<num_blocks, num_threads>>>(this->layer_errors[i]->data, averages->data, this->layer_errors[i]->height,this->layer_errors[i]->width);

                struct Matrix *delta_weights = create_cuda_matrix(this->topology[i] + 1, this->topology[i + 1]);
            }
        }
        

        void back_propagate(float learning_rate)
        {
            calculate_hidden_errors();
            adjust_weights(learning_rate);
        }
        
        void train(struct Matrix *input, struct Matrix *ground_truth,int epochs, float learning_rate)
        {
            int num_samples = input->height;
            // create array to store inputs and ground truth on gpu and populate it
            struct Matrix *d_inputs, *d_ground_truth;
            d_inputs = send_matrix_to_gpu(input);
            d_ground_truth = send_matrix_to_gpu(ground_truth);

            this->error = create_cuda_matrix(num_samples, this->num_outputs);

            for (int i = 0; i < this->num_layers - 1; i++)
            {
                this->layer_errors[i] = create_cuda_matrix(num_samples,this->topology[i] + 1);
            }

            // // create arrays to store output of each layer on gpu
            for (int i = 1; i < num_layers; i++)
            {
                this->layer_outputs[i-1] = create_cuda_matrix(num_samples, this->topology[i]);
                // cudaMalloc(&(layer_outputs[i-1]), (topology[i]+1) * num_samples * sizeof(float));
            }


            printf("Training Neural Network\n");
            for (int i = 0; i < epochs; i++)
            {
                printf("Epoch %i\n", i);
                forward_propagate(d_inputs);
                calculate_output_error(d_ground_truth);
                // print_matrix(this->error);
                back_propagate(learning_rate);
            }

            // print results
            // float *temp = (float *) malloc(this->num_outputs * num_samples * sizeof(float));
            // cudaMemcpy(temp, this->layer_outputs[this->num_layers - 2],  this->num_outputs * num_samples * sizeof(float), cudaMemcpyDeviceToHost);
            // for (int i = 0; i < num_samples; i++)
            // {
            //     for (int j = 0; j < this->num_outputs; j++)
            //     {
            //         printf("%f ",temp[this->num_outputs * i + j] );
            //     }
            //     printf("\n");
            //     for (int j = 0; j < this->num_outputs; j++)
            //     {
            //         printf("%f ",this->error[this->num_outputs * i + j] );
            //     }
            //     printf("\n");
            // }

            // float *temp = (float *) malloc((this->num_inputs + 1) * num_samples * sizeof(float));
            // cudaMemcpy(temp, this->layer_errors[0],  (this->num_inputs + 1) * num_samples * sizeof(float), cudaMemcpyDeviceToHost);
            // for (int i = 0; i < num_samples; i++)
            // {
            //     for (int j = 0; j < this->num_inputs + 1; j++)
            //     {
            //         printf("%f ",temp[(this->num_inputs + 1) * i + j] );
            //     }
            //     printf("\n");
            // }


            // free GPU memory
            for (int i = 0; i < num_layers -1; i++)
            {
                cudaFree(layer_outputs[i]);
            }
            cudaFree(d_inputs);
            cudaFree(d_ground_truth);
            cudaFree(this->d_error);
            free(this->error);

        }

    
};


#endif