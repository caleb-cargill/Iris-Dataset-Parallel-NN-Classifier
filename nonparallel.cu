/*
 CS5168/CS6068 - Parallel Computing

 Final Project - Iris Dataset Parallel NN Classifier
 November, 2021

 Team Members:
 - Ben Elfner
 - Caleb Cargill

 File Description:
 - Non-parallized version of the NN classifier, classifying the Iris Dataset
*/

#include <sstream>
#include <string>
#include <iostream>
#include "cuda.h"
#include <iostream>
#include <iomanip>
#include <string>							
#include <math.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <vector>
#include "Iris.h"
#include <random>
#include "gputimer.h"

// General Constants
static const int DATASET_SIZE = 150; // number of rows in the dataset
static const int TRAINING_SIZE = DATASET_SIZE * 2 / 3; // number of rows in dataset to use as training
static const int TESTING_SIZE = DATASET_SIZE / 3; // number of rows in the dataset ot use for testing
static const int NUM_FEATURES = 5; // number of features in the dataset; 4 input, 1 output
static const std::string DATASET_FILE_NAME = "iris.csv"; // file name that the dataset is stored in

// Training Constants
static const int EPOCHS = 50; // number of iterations to train with
static const double LEARNING_RATE = 0.001; // learning rate for updating weights

// Dataset
static Iris iris_dataset[DATASET_SIZE]; // variable to store dataset in
static Iris training_data[TRAINING_SIZE]; // variable to store training data
static Iris testing_data[TESTING_SIZE]; // variable to store testing data
static float weights[NUM_FEATURES]; // weights for single layer network
static int weight_updates = 0; // used to track the number of times the weights array is updated

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
	// Split Dataset into Training and Testing Data
	int train_count, test_count = 0;
	for (int i = 0; i < DATASET_SIZE; i++) {
		if (i % 3 == 0) {
			testing_data[test_count] = iris_dataset[i];
			test_count++;
		}
		else {
			training_data[train_count] = iris_dataset[i];
			train_count++;
		}
	}
}

// Used to update weight array after predictions
void update_weights(Iris iris, float prediction) {
	if (iris.SpeciesVal - prediction == 0)
		return;

	weights[0] += LEARNING_RATE * (iris.SpeciesVal - prediction) * iris.SepalLength;
	weights[1] += LEARNING_RATE * (iris.SpeciesVal - prediction) * iris.SepalWidth;
	weights[2] += LEARNING_RATE * (iris.SpeciesVal - prediction) * iris.PetalLength;
	weights[3] += LEARNING_RATE * (iris.SpeciesVal - prediction) * iris.PetalWidth;
	weights[4] += LEARNING_RATE * (iris.SpeciesVal - prediction) * iris.SpeciesVal;

	weight_updates += 1;
}

// Uses weights to predict flower type for passed in Iris data
// Returns the double value prediction
// Using a single layer perceptron network
float predict(Iris iris) {
	float prediction = iris.dot(weights);

	// 'round' prediction
	if (prediction < 1.0) {
		prediction = 0.0;
	}
	else if (prediction > 1.0 && prediction < 2.0) {
		prediction = 1.0;
	}
	else {
		prediction = 2.0;
	}

	return prediction;
}

// Used for training, calls predict method for Iris data and then updates weights if needed 
void train() {
	float training_accuracy = 0.0;

	// Initialize weights (4 Input Features, 1 Bias)
	for (int i = 0; i < NUM_FEATURES; i++)
		weights[i] = 1;

	for (int i = 0; i < EPOCHS; i++) {
		float prediction = 0.0;
		float accuracy = 0.0;

		for (int j = 0; j < TRAINING_SIZE; j++) {
			prediction = predict(training_data[j]);

			if (prediction == training_data[j].SpeciesVal) {
				accuracy += 1.0;
			}
			else {
				accuracy += 0.0;
			}

			update_weights(training_data[j], prediction);
		}

		// Update accuracies
		accuracy = accuracy / TRAINING_SIZE;
		training_accuracy += accuracy;
	}

	training_accuracy = training_accuracy / EPOCHS;
	printf("\nAverage Training Accuracy: %f", training_accuracy);
}

void test() {
	float prediction = 0.0;
	float accuracy = 0.0;

	for (int j = 0; j < TESTING_SIZE; j++) {
		prediction = predict(testing_data[j]);

		if (prediction == testing_data[j].SpeciesVal) {
			accuracy += 1.0;
		}
		else {
			accuracy += 0.0;
		}
	}

	accuracy = accuracy / TESTING_SIZE;

	printf("\nAverage Testing Accuracy: %f", accuracy);
}

int main(void)
{
	// Initialize and start timer
	GpuTimer timer;
	timer.Start();

	// Initialize dataset
	init_dataset();

	// Split dataset into training and testing data
	split_dataset();

	// Train on training data
	train();

	// Test on testing data
	test();

	// Print some statistics
	timer.Stop();
	printf("\nTotal Weight Updates: %i", weight_updates);
	printf("\nTime Elapsed: %g ms\n", timer.Elapsed());

	return 0;
}