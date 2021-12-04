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
	printf("\nTime Elapsed: %g ms\n", timer.Elapsed());

	return 0;
}