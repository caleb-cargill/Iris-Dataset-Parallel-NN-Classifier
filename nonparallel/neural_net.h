#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <random>
#include <algorithm>
#include "../Iris.h"
#include "cuda.h"
#include <math.h>
#include "../data_manip.h"

// General Constants
static const int DATASET_SIZE = 150; // number of rows in the dataset
static const int TRAINING_SIZE = 50; // number of rows in dataset to use as training
static const int TESTING_SIZE = 100; // number of rows in the dataset ot use for testing
static const int NUM_FEATURES = 5; // number of features in the dataset; 4 input, 1 output
static const int NUM_LAYERS = 3; // number of layers in the neural network
static const std::string DATASET_FILE_NAME = "iris.csv"; // file name that the dataset is stored in
static const int TOPOLOGY[NUM_LAYERS] = { 4, 3, 3 }; // neural network structure
static const int NUM_NODES = 10;

class NeuralNetwork {
private:
	int epochs = 0;
	float learning_rate = 0;
	float y[TRAINING_SIZE];
	float y_history[TRAINING_SIZE * 3];
	float prev_grad[TRAINING_SIZE];
	float d_x[TRAINING_SIZE];
	float d_active[TRAINING_SIZE];
	int feed_count = 0;

	// Dataset
	Iris iris_dataset[DATASET_SIZE]; // variable to store dataset in
	Iris training_data[TRAINING_SIZE]; // variable to store training data
	Iris testing_data[TESTING_SIZE]; // variable to store testing data
	float weights[NUM_NODES + NUM_LAYERS]; // weights for network	

	// Method to initialize the Iris dataset
	void init_dataset() {
		std::ifstream infile(DATASET_FILE_NAME);
		std::string line, word;
		int i = 0;
		float raw_dataset[DATASET_SIZE][NUM_FEATURES];

		while (std::getline(infile, line))
		{
			std::stringstream s(line);
			int j = 0;
			while (getline(s, word, ',')) {
				raw_dataset[i][j] = std::stof(word);
				j++;
			}

			iris_dataset[i] = *(new Iris(raw_dataset[i]));
			i++;
		}
	}

	// splits the data set into training and testing data
	void split_dataset() {
		// Get randomized int array (0-149) for randomizing training and testing data split
		int* sample_array = (int*)malloc(DATASET_SIZE * sizeof(int));
		create_sample_array(sample_array, DATASET_SIZE);

		// Split Dataset into Training and Testing Data
		int train_count, test_count = 0;
		for (int i = 0; i < DATASET_SIZE; i++) {
			if (i % 3 == 0) {
				training_data[train_count] = iris_dataset[sample_array[i]];
				train_count++;
			}
			else {
				testing_data[test_count] = iris_dataset[sample_array[i]];
				test_count++;
			}
		}
	}

	// initializes the weights for each layer
	void initialize_weights() {
		std::default_random_engine generator;
		int weight_num = 0;

		// iterate over layers to initialize weights
		for (int i = 0; i < NUM_LAYERS - 1; i++) {
			// weights range from -1/sqrt(n) to 1/sqrt(n) where n is the number of nodes in the previous layer
			std::uniform_real_distribution<float> dis(-1.0 / sqrt(TOPOLOGY[i]), 1.0 / sqrt(TOPOLOGY[i]));

			// iterate over weights and initialize for this layer (add one to include a bias value)
			for (int j = 0; j < TOPOLOGY[i] + 1; j++) {
				weights[weight_num] = dis(generator);
				weight_num++;
			}
		}
	}

	// activation function 
	float sigmoid(float input) {
		return 1.0 / (1.0 + exp(-1.0 * input));
	}

	// derivative of activation function. Input must be output of sigmoid function
	float sigmoid_deriv(float input) {
		return input * (1.0 - input);
	}
public:
	float training_accuracy = 0.0;
	float actual_accuracy = 0.0;

	// Constructor
	NeuralNetwork(int epochs, float learning_rate) {
		this->epochs = epochs;
		this->learning_rate = learning_rate;

		// Initialize dataset
		init_dataset();

		// Split dataset into training and testing data
		split_dataset();

		// Initialize the weights
		initialize_weights();
	}

	void calculate_prev_grad() {
		for (int i = 0; i < TRAINING_SIZE; i++) {
			prev_grad[i] = -1.0 * (training_data[i].SpeciesVal - y[i]);
		}
	}

	void calculate_d_activ(int layer_num) {
		for (int i = 0; i < TRAINING_SIZE; i++) {
			d_active[i] = prev_grad[i] * sigmoid_deriv(y_history[TRAINING_SIZE * layer_num + i]);
		}
	}

	void backpropagation() {

		// iterate over weights in reverse order
		for (int i = NUM_LAYERS - 1; i >= 0; i--) {
			if (i == NUM_LAYERS - 1) {
				calculate_prev_grad();
			}
			else {
				for (int j = 0; j < TRAINING_SIZE; j++) {
					prev_grad[j] = d_x[j];
				}
			}

			calculate_d_activ(i);

			// Calculate delta weights
			float d_weight[TOPOLOGY[i]];
			if (i > 0) {
				// use y history
				for (int j = 0; j < TOPOLOGY[i]; j++) {
					d_weight[j] = 0.0;
					for (int k = 0; k < TRAINING_SIZE; k++) {
						d_weight[j] += y_history[(TRAINING_SIZE * (i - 1)) + k] * d_active[k];
					}
				}
			}
			else {
				// use inputs
				for (int j = 0; j < TOPOLOGY[i]; j++) {
					for (int k = 0; k < TRAINING_SIZE; k++) {
						d_weight[j] += training_data[k].SepalLength * d_active[k];
						d_weight[j] += training_data[k].SepalWidth * d_active[k];
						d_weight[j] += training_data[k].PetalLength * d_active[k];
						d_weight[j] += training_data[k].PetalWidth * d_active[k];
					}
				}
			}

			// calculate delta bias
			float d_bias = 0.0;
			for (int j = 0; j < TRAINING_SIZE; j++) {
				d_bias += d_active[j];
			}

			// calculate next grad
			for (int j = 0; j < TRAINING_SIZE; j++) {
				d_x[j] = 0.0;
				for (int k = 0; k < TOPOLOGY[i]; k++) {
					d_x[j] += d_active[j] * weights[k];
				}
			}

			// update weights and bias for this layer
			// indexes 0-4 is layer 1, 5-8 is layer 2, 9-12 is layer 3
			if (i == 2) {
				weights[9] += (-1.0 * learning_rate * d_weight[0]);
				weights[10] += (-1.0 * learning_rate * d_weight[1]);
				weights[11] += (-1.0 * learning_rate * d_weight[2]);
				weights[12] += (-1.0 * learning_rate * d_bias);
			}
			else if (i == 1) {
				weights[5] += (-1.0 * learning_rate * d_weight[0]);
				weights[6] += (-1.0 * learning_rate * d_weight[1]);
				weights[7] += (-1.0 * learning_rate * d_weight[2]);
				weights[8] += (-1.0 * learning_rate * d_bias);
			}
			else if (i == 0) {
				weights[0] += (-1.0 * learning_rate * d_weight[0]);
				weights[1] += (-1.0 * learning_rate * d_weight[1]);
				weights[2] += (-1.0 * learning_rate * d_weight[2]);
				weights[3] += (-1.0 * learning_rate * d_weight[3]);
				weights[4] += (-1.0 * learning_rate * d_bias);
			}
		}
	}

	/// <summary>
	/// Calculates the model output and stores each layer's output for use in backpropagation
	/// </summary>
	void feed_forward_training() {
		int weight_count = 0;
		float input = 0.0;

		// first layer
		for (int i = 0; i < TRAINING_SIZE; i++) {
			input += training_data[i].SepalLength * weights[weight_count];
			input += training_data[i].SepalWidth * weights[weight_count + 1];
			input += training_data[i].PetalLength * weights[weight_count + 2];
			input += training_data[i].PetalWidth * weights[weight_count + 3];
			input += weights[weight_count + 4];
			y[i] = sigmoid(input);
			y_history[i] = y[i];
		}

		// second layer
		weight_count += 5;
		input = 0.0;
		for (int i = 0; i < TRAINING_SIZE; i++) {
			input += y[i] * weights[weight_count];
			input += y[i] * weights[weight_count + 1];
			input += y[i] * weights[weight_count + 2];
			input += weights[weight_count + 3];
			y[i] = sigmoid(input);
			y_history[TRAINING_SIZE + i] = y[i];
		}

		// third layer
		weight_count += 4;
		input = 0.0;
		for (int i = 0; i < TRAINING_SIZE; i++) {
			input += y[i] * weights[weight_count];
			input += y[i] * weights[weight_count + 1];
			input += y[i] * weights[weight_count + 2];
			input += weights[weight_count + 3];
			y[i] = sigmoid(input);
			y_history[(TRAINING_SIZE * 2) + i] = y[i];
		}
	}

	/// <summary>
	/// Calculates the model output and stores each layer's output for use in backpropagation
	/// </summary>
	void feed_forward() {
		int weight_count = 0;
		float input = 0.0;

		// first layer
		for (int i = 0; i < TESTING_SIZE; i++) {
			input += testing_data[i].SepalLength * weights[weight_count];
			input += testing_data[i].SepalWidth * weights[weight_count + 1];
			input += testing_data[i].PetalLength * weights[weight_count + 2];
			input += testing_data[i].PetalWidth * weights[weight_count + 3];
			input += weights[weight_count + 4];
			y[i] = sigmoid(input);
			y_history[i] = y[i];
		}

		// second layer
		weight_count += 5;
		input = 0.0;
		for (int i = 0; i < TESTING_SIZE; i++) {
			input += y[i] * weights[weight_count];
			input += y[i] * weights[weight_count + 1];
			input += y[i] * weights[weight_count + 2];
			input += weights[weight_count + 3];
			y[i] = sigmoid(input);
			y_history[TESTING_SIZE + i] = y[i];
		}

		// third layer
		weight_count += 4;
		input = 0.0;
		for (int i = 0; i < TESTING_SIZE; i++) {
			input += y[i] * weights[weight_count];
			input += y[i] * weights[weight_count + 1];
			input += y[i] * weights[weight_count + 2];
			input += weights[weight_count + 3];
			y[i] = sigmoid(input);
			y_history[(TESTING_SIZE * 2) + i] = y[i];
		}
	}

	/// <summary>
	/// Trains the neural network to produce a set of weights + biases
	/// Prints training accuracy at the end of training
	/// </summary>
	void train() {
		training_accuracy = 0.0;

		// Iterate (number of iterations)
		for (int i = 0; i < epochs; i++) {
			feed_forward_training();

			float loss = 0.0;
			for (int j = 0; j < TRAINING_SIZE; j++) {
				loss += (training_data[j].SpeciesVal - y[j]) * (training_data[j].SpeciesVal - y[j]);
			}
			loss = loss / TRAINING_SIZE;

			if (i % 10 == 0) {
				printf("\nLoss for Epoch #%i: %f", i, loss);
			}

			backpropagation();
			training_accuracy += (1 - loss);
		}

		training_accuracy = training_accuracy / epochs;
		printf("\nDone Training. Training Accuracy: %f", training_accuracy);

	}

	/// <summary>
	/// Tests the already trained neural network on testing data
	/// Prints the testing accuracy at the end of testing
	/// </summary>
	void test() {
		actual_accuracy = 0.0;

		feed_forward();

		float loss = 0.0;
		for (int j = 0; j < TESTING_SIZE; j++) {
			loss += (testing_data[j].SpeciesVal - y[j]) * (testing_data[j].SpeciesVal - y[j]);
		}
		loss = loss / TESTING_SIZE;

		actual_accuracy += (1 - loss);

		printf("\nDone Testing. Actual (Testing) Accuracy: %f", actual_accuracy);
	}
};

#endif