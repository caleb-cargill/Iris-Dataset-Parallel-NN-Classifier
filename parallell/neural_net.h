#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <random> 
#include <algorithm>
#include "mat_mult.h"
#include "cuda.h"
// class NeuralNetwork {
//     public:
//         NeuralNetwork::NeuralNetwork(int *topology, int num_layers);
//         NeuralNetwork::~NeuralNetwork();
//         // void train(double *input, double *output, int epochs);
//         // void test(double *input, double *output);
//         // void save(char *filename);
//         // void load(char *filename);
//     private:
//         int *topology;
//         int num_inputs;
//         int num_outputs;
//         int num_hidden_layers;
//         int output_size;
//         int *neurons;
// };

class NeuralNetwork {
    private:
        int *topology;
        int num_inputs;
        int num_outputs;
        int num_hidden_layers;
        int output_size;
        int num_layers;
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

        void train(float *input, float *ground_truth,int epochs, float learning_rate,int num_samples)
        {
            // create array to store inputs and ground truth on gpu and populate it
            float *d_inputs, *d_ground_truth;
            cudaMalloc(&d_inputs, (this->num_inputs +1) * num_samples * sizeof(float));
            cudaMalloc(&d_ground_truth, this->num_outputs * num_samples * sizeof(float));
            cudaMemcpy(d_inputs, input, (this->num_inputs + 1) * num_samples * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ground_truth, ground_truth, this->num_outputs * num_samples * sizeof(float), cudaMemcpyHostToDevice);


            // create arrays to store output of each layer on gpu
            this->layer_outputs = new float*[num_layers-1];
            for (int i = 1; i < num_layers; i++)
            {
                cudaMalloc(&(layer_outputs[i-1]), topology[i] * num_samples * sizeof(float));
            }


            printf("Training Neural Network\n");
            for (int i = 0; i < epochs; i++)
            {
                printf("Epoch %i\n", i);
                forward_propagate(d_inputs,num_samples);

                // back_prop(output, learning_rate);
            }

            // print results
            float *temp = (float *) malloc(this->num_outputs * num_samples * sizeof(float));
            cudaMemcpy(temp, this->layer_outputs[0],  this->num_outputs * num_samples * sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < num_samples; i++)
            {
                for (int j = 0; j < this->num_outputs; j++)
                {
                    printf("%f ",temp[this->num_outputs * i + j] );
                }
                printf("\n");
            }

            // free GPU memory
            for (int i = 0; i < num_layers -1; i++)
            {
                cudaFree(layer_outputs[i]);
            }
            cudaFree(d_inputs);
            cudaFree(d_ground_truth);

        }

    
};


#endif