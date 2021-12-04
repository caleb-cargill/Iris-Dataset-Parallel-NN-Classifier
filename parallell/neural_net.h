#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <random> 
#include <algorithm>
#include "mat_mult.h"
#include "matrix.h"
#include "cuda.h"


class NeuralNetwork {
    private:
        int *topology;
        int num_inputs, num_outputs;
        int num_hidden_layers, num_layers;
        float accuracy;
        struct Matrix *error, *d_error;
        struct Matrix *d_inputs, *d_ground_truth;
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

                struct Matrix * temp = create_matrix(topology[i] + 1, topology[i + 1]);

                for(int y = 0; y < temp->height* temp->width; y++)
                {
                    temp->data[y] = distribution(generator);
                }
                this->weights[i] = send_matrix_to_gpu(temp);
                matrix_destroy(temp);
            }

        }
        NeuralNetwork::~NeuralNetwork()
        {
            printf("Destroying Neural Network\n");
            for (int i = 0; i < num_layers - 1; i++)
            {
                matrix_destroy(this->weights[i]);
                matrix_destroy(this->layer_errors[i]);
                matrix_destroy(this->layer_outputs[i]);
            }
            delete[] this->weights;
            delete[] this->topology;
            delete[] this->layer_errors;
            delete[] this->layer_outputs;
            matrix_destroy(this->d_inputs);
            matrix_destroy(this->d_ground_truth);
            matrix_destroy(this->d_error);
            matrix_destroy(this->error);

        }

        void forward_propagate()
        {
            int i = 0;
            nn_mat_mul(this->d_inputs, this->weights[i], this->layer_outputs[i]);
            i++;
            for (; i < this->num_layers - 1; i++)
            {
                nn_mat_mul(this->layer_outputs[i-1], this->weights[i], this->layer_outputs[i]);
            }
        }

        void calculate_output_error()
        {
            this->d_error = create_cuda_matrix(this->d_ground_truth->height, this->d_ground_truth->width);

            int num_threads = this->d_ground_truth->height;
            int num_blocks = 1;
            if (this->d_ground_truth->height > 512)
            {
                num_threads = 512;
                num_blocks = ceil(float(this->d_ground_truth->height) / 512.0);
            }

            calculate_output_error_kernal<<<num_blocks, num_threads>>>(this->layer_outputs[this->num_layers - 2]->data, this->d_ground_truth->data, this->d_error->data,this->num_outputs,this->d_ground_truth->height);
            this->error = retrieve_matrix_from_gpu(this->d_error);
        }

        void calculate_hidden_errors()
        {
            int num_samples =  this->layer_errors[0]->height;
            for (int x = this->num_layers - 2; x >= 0; x--)
            {
                int num_threads = num_samples;
                int num_blocks = 1;
                if (num_samples > 512)
                {
                    num_threads = 512;
                    num_blocks = ceil(float(num_samples) / 512.0);
                }
                if (x == this->num_layers - 2)
                {
                    calculate_hidden_error_kernal<<<num_blocks, num_threads>>>(this->layer_errors[x]->data, this->weights[x]->data, this->d_error->data, this->layer_outputs[x]->data ,this->topology[x] + 1, this->topology[x + 1], num_samples);
                }
                else
                {
                    calculate_hidden_error_kernal<<<num_blocks, num_threads>>>(this->layer_errors[x]->data, this->weights[x]->data, this->layer_errors[x+1]->data, this->layer_outputs[x]->data ,this->topology[x] + 1, this->topology[x + 1], num_samples);
                }
            }

        }

        void adjust_weights(float learning_rate)
        {
            // formula 
            // weight = weight - learning_rate * error * input

            for (int i = 0; i < this->num_layers - 1; i++)
            {   
                // printf("i: %i\n",i);
                //  error * input
                struct Matrix * temp  = create_cuda_matrix(this->layer_errors[i]->height, this->layer_errors[i]->width);
                if (i == 0)
                {
                    element_wise_multiplication(this->layer_errors[i], this->d_inputs, temp, true);
                }
                else
                {
                    element_wise_multiplication(this->layer_errors[i], this->layer_outputs[i - 1], temp, true);
                }

                // average error over all inputs
                struct Matrix *averages = create_cuda_matrix(1, this->topology[i] + 1);
                int num_threads = this->topology[i] + 1;
                int num_blocks = 1;
                if (num_threads > 512)
                {
                    num_threads = 512;
                    num_blocks = ceil(float(this->topology[i] + 1) / 512.0);
                }
                average_columns_kernal<<<num_blocks, num_threads>>>(temp->data, averages->data, this->layer_errors[i]->height,this->layer_errors[i]->width);

                // weight = weight - learning_rate * averages
                struct Matrix *delta_weights = create_cuda_matrix(this->topology[i] + 1, this->topology[i + 1]);
                
                dim3 threadsPerBlock(weights[i]->width, weights[i]->height);
                dim3 blocksPerGrid(1, 1);
                if (weights[i]->width * weights[i]->height > 512){
                    threadsPerBlock.x = 512;
                    threadsPerBlock.y = 512;
                    blocksPerGrid.x = ceil(double(weights[i]->width)/double(threadsPerBlock.x));
                    blocksPerGrid.y = ceil(double(weights[i]->height)/double(threadsPerBlock.y));
                }
                update_weights_kernal<<<blocksPerGrid, threadsPerBlock>>>(weights[i]->data, averages->data, weights[i]->height, weights[i]->width, learning_rate);
                matrix_destroy(temp);
                matrix_destroy(averages);
                matrix_destroy(delta_weights);
            }
        }
        

        void back_propagate(float learning_rate)
        {
            calculate_hidden_errors();
            adjust_weights(learning_rate);
        }

        void calculate_accuracy()
        {
            struct Matrix * output = retrieve_matrix_from_gpu(this->layer_outputs[this->num_layers - 2]);
            struct Matrix * ground_truth = retrieve_matrix_from_gpu(this->d_ground_truth);
            int num_correct = 0;
            for (int i = 0; i < this->d_ground_truth->height; i++)
            {
                int max_index = 0;
                float max_value = output->data[i * output->width];
                for (int j = 1; j < output->width; j++)
                {
                    if (output->data[i * output->width + j] > max_value)
                    {
                        max_value = output->data[i * output->width + j];
                        max_index = j;
                    }
                    
                }
                if (ground_truth->data[i * output->width + max_index] == 1)
                {
                    num_correct++;
                }
            }
            this->accuracy = float(num_correct) / float(this->d_ground_truth->height);

        }
        
        void train(struct Matrix *input, struct Matrix *ground_truth,int epochs, float learning_rate)
        {
            int num_samples = input->height;
            // create array to store inputs and ground truth on gpu and populate it
            
            this->d_inputs = send_matrix_to_gpu(input);
            this->d_ground_truth = send_matrix_to_gpu(ground_truth);

            this->error = create_cuda_matrix(num_samples, this->num_outputs);

            for (int i = 0; i < this->num_layers - 1; i++)
            {
                this->layer_errors[i] = create_cuda_matrix(num_samples,this->topology[i] + 1);
            }

            // // create arrays to store output of each layer on gpu
            for (int i = 1; i < num_layers; i++)
            {
                this->layer_outputs[i-1] = create_cuda_matrix(num_samples, this->topology[i]);
            }


            printf("Training Neural Network\n");
            for (int i = 0; i < epochs; i++)
            {
                forward_propagate();
                calculate_output_error();
                back_propagate(learning_rate);

                if (i % 10 == 0)
                {
                    printf("Epoch %i\n", i);
                    calculate_accuracy();
                    printf("Accuracy: %f\n", this->accuracy);
                }

            }

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