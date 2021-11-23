/*
 CS5168/CS6068 - Parallel Computing

 Final Project - Iris Dataset Parallel NN Classifier
 November, 2021

 Team Members:
 - Ben Elfner
 - Caleb Cargill

 File Description:
 - Class to store Iris information
*/

#pragma once
#include <string>

class Iris {
private:
	static const int NUM_FEATURES = 5;
public:
	double SepalLength;
	double SepalWidth;
	double PetalLength;
	double PetalWidth;
	double SpeciesVal;
	std::string Species;
	bool IsSetosa;

	// Default Constructor
	Iris() {
		SepalLength = 0;
		SepalWidth = 0;
		PetalLength = 0;
		PetalWidth = 0;
		SpeciesVal = 0;
		Species = "";
		IsSetosa = false;
	}

	// Constructor
	Iris(float data[NUM_FEATURES]) {
		SepalLength = data[0];
		SepalWidth = data[1];
		PetalLength = data[2];
		PetalWidth = data[3];
		SpeciesVal = data[4];
		int spec = data[4];
		switch (spec) {
		case 0:
			Species = "setosa";
			IsSetosa = true;
		case 1:
			Species = "versicolor";
			IsSetosa = false;
		case 2:
			Species = "virginica";
			IsSetosa = false;
		}
	}

	// Method to Print Iris Data
	void Print() {
		printf("Sepal Length: %f\n", SepalLength);
		printf("Sepal Width: %f\n", SepalWidth);
		printf("Petal Length: %f\n", PetalLength);
		printf("Petal Width: %f\n", PetalWidth);
		printf("Species: %s\n", Species);
	}

	// Computes dot product between Iris features and weight array
	double dot(float(&w)[5]) {
		return w[0] * SepalLength + w[1] * SepalWidth + w[2] * PetalLength + w[3] * PetalWidth + w[4];
	}
};
