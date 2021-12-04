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
#include "../Iris.h"
#include <random>
#include "../gputimer.h"
#include "neural_net.h"

// Training Constants
static const int EPOCHS = 50; // number of iterations to train with
static const double LEARNING_RATE = 0.001; // learning rate for updating weights

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

	NeuralNetwork nn = *(new NeuralNetwork(EPOCHS, LEARNING_RATE));

	nn.train();

	nn.test();

	// Print some statistics
	timer.Stop();
	printf("\nTotal Weight Updates: %i", weight_updates);
	printf("\nTime Elapsed: %g ms\n", timer.Elapsed());

	return 0;
}