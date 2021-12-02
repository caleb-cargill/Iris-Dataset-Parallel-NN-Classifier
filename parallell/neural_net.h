#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <random> 
#include <algorithm>
#include "mat_mult.h"
#include "cuda.h"

__global__ void calculate_output_error_kernal(float* output, float *ground_truth, float *error, int width,int total)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < total)
    {
        for (int i = 0; i < width; i ++)
        {
            error[idx*width+i] = (output[idx*width+i] - ground_truth[idx*width+i])*(1-output[idx*width+i])*output[idx*width+i];
        }
    }
}

__global__ void calculate_hidden_error_kernal(float* error, float *weights, float *next_error, int current_width, int next_width,int total)
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
            error[idx*current_width+i] = acc*(1-error[idx*current_width+i])*error[idx*current_width+i];
        }
    }
}


class NeuralNetwork {
    private:
        int *topology;
        int num_inputs;
        int num_outputs;
        int num_hidden_layers;
        int num_layers;
        float *error;
        float *d_error; // error stored on GPU
        float **layer_errors;
        float **weights;
        float **layer_outputs;
    public:
        NeuralNetwork(int *topology, int num_layers)
        {
            printf("Constructing Neural Network\n");
            this->num_layers = num_layers;
            this->topology = topology;
            this->num_inputs = topology[0];
            this->num_outputs = topology[num_layers - 1];
            this->num_hidden_layers = num_layers - 2;

            this->weights = new float*[num_layers - 1];
            this->layer_errors = new float*[num_layers - 1];
            std::default_random_engine generator;
            // create weights matrix for each layer
            for (int i = 0; i < num_layers - 1; i++)
            {
                // weights range from -1/sqrt(n) to 1/sqrt(n) where n is the number of nodes in the previous layer
                std::uniform_real_distribution<float> distribution(-1.0/sqrt(topology[i]),1.0/sqrt(topology[i]));

                printf("num weights %i\n",(topology[i] + 1) * topology[i + 1]);

                float *temp = (float *) malloc((topology[i] + 1) * topology[i + 1] * sizeof(float));
                for(int y = 0; y < (topology[i] + 1) * topology[i + 1]; y++)
                {
                    temp[y] = distribution(generator);
                }

                // create GPU array and populate it
                cudaMalloc(&(this->weights[i]), (topology[i] + 1) * topology[i + 1] * sizeof(float));
                cudaMemcpy(this->weights[i], temp, (topology[i] + 1) * topology[i + 1] * sizeof(float), cudaMemcpyHostToDevice);
                
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

        void forward_propagate(float *input,int num_inputs)
        {
            int i = 0;
            nn_mat_mul(input, this->weights[i], this->layer_outputs[i], num_inputs, this->topology[i], this->topology[i + 1]);
            i++;
            printf("i: %i\n",i);
            for (; i < this->num_layers - 1; i++)
            {
                nn_mat_mul(this->layer_outputs[i-1], this->weights[i], this->layer_outputs[i] ,num_inputs, this->topology[i] + 1, this->topology[i + 1]);
                printf("i: %i\n",i);
            }
        }

        void calculate_output_error(float* ground_truth, int num_samples)
        {
            int num_threads = num_samples;
            int num_blocks = 1;
            if (num_samples > 512)
            {
                num_threads = 512;
                num_blocks = ceil(float(num_samples) / 512.0);
            }

            // float *t1, *t2, *t3;
            // t1 = (float *) malloc(num_samples * this->num_outputs * sizeof(float));
            // t2 = (float *) malloc(num_samples * this->num_outputs * sizeof(float));
            // t3 = (float *) malloc(num_samples * this->num_outputs * sizeof(float));
            // cudaMemcpy(t1, this->layer_outputs[this->num_layers - 2], num_samples * this->num_outputs * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(t2, ground_truth, num_samples * this->num_outputs * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(t3, this->d_error, num_samples * this->num_outputs * sizeof(float), cudaMemcpyDeviceToHost);

            // for (int i = 0; i < num_samples; i++)
            // {
            //     for (int j = 0; j < this->num_outputs; j++)
            //     {
            //         printf("%f ",t1[i*this->num_outputs + j]);
            //     }
            //     printf("\n");
            // }
            // printf("\n");
            // for (int i = 0; i < num_samples; i++)
            // {
            //     for (int j = 0; j < this->num_outputs; j++)
            //     {
            //         printf("%f ",t2[i*this->num_outputs + j]);
            //     }
            //     printf("\n");
            // }
            // printf("\n");
            // for (int i = 0; i < num_samples; i++)
            // {
            //     for (int j = 0; j < this->num_outputs; j++)
            //     {
            //         printf("%f ",t3[i*this->num_outputs + j]);
            //     }
            //     printf("\n");
            // }
            // printf("\n");
            calculate_output_error_kernal<<<num_blocks, num_threads>>>(this->layer_outputs[this->num_layers - 2], ground_truth, this->d_error,this->num_outputs,num_samples);
            cudaMemcpy(this->error, this->d_error, num_samples * this->num_outputs * sizeof(float), cudaMemcpyDeviceToHost);
        }

        void calculate_hidden_errors(int num_samples)
        {
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
                if (x == this->num_layers - 1)
                {
                    calculate_hidden_error_kernal<<<num_blocks, num_threads>>>(this->layer_errors[x], this->weights[x], this->d_error,  this->topology[x] + 1, this->topology[x + 1], num_samples);
                   
                }
                else
                {
                    calculate_hidden_error_kernal<<<num_blocks, num_threads>>>(this->layer_errors[x], this->weights[x], this->layer_errors[x+1],  this->topology[x] + 1, this->topology[x + 1], num_samples);
                }
                float *temp = (float *) malloc((this->topology[x] + 1) * num_samples * sizeof(float));
                cudaMemcpy(temp, this->layer_errors[x],  (this->topology[x] + 1) * num_samples * sizeof(float), cudaMemcpyDeviceToHost);
                for (int i = 0; i < num_samples; i++)
                {
                    for (int j = 0; j < (this->topology[x] + 1); j++)
                    {
                        printf("%f ",temp[i*(this->topology[x] + 1) + j]);
                    }
                    printf("\n");
                }
            }

        }
            

        void back_propagate(float learning_rate,int num_samples)
        {
            calculate_hidden_errors(num_samples);
        }

        void train(float *input, float *ground_truth,int epochs, float learning_rate,int num_samples)
        {
            // create array to store inputs and ground truth on gpu and populate it
            float *d_inputs, *d_ground_truth;
            cudaMalloc(&d_inputs, (this->num_inputs +1) * num_samples * sizeof(float));
            cudaMalloc(&d_ground_truth, this->num_outputs * num_samples * sizeof(float));
            
            cudaMemcpy(d_inputs, input, (this->num_inputs + 1) * num_samples * sizeof(float), cudaMemcpyHostToDevice); // + 1 for type column
            cudaMemcpy(d_ground_truth, ground_truth, this->num_outputs * num_samples * sizeof(float), cudaMemcpyHostToDevice);

            this->error = (float *) malloc(num_samples * this->num_outputs * sizeof(float));
            cudaMalloc(&(this->d_error), num_samples * this->num_outputs * sizeof(float));

            for (int i = 0; i < this->num_layers - 1; i++)
            {
                cudaMalloc(&(this->layer_errors[i]), (this->topology[i] + 1) * num_samples * sizeof(float));
            }


            // create arrays to store output of each layer on gpu
            this->layer_outputs = new float*[num_layers-1];
            for (int i = 1; i < num_layers; i++)
            {
                cudaMalloc(&(layer_outputs[i-1]), (topology[i]+1) * num_samples * sizeof(float));
            }


            printf("Training Neural Network\n");
            for (int i = 0; i < epochs; i++)
            {
                printf("Epoch %i\n", i);
                forward_propagate(d_inputs, num_samples);
                calculate_output_error(d_ground_truth,num_samples);
                back_propagate(learning_rate,num_samples);
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