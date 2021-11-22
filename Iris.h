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
	std::string Species;
	bool IsSetosa;

	// Default Constructor
	Iris() {
		SepalLength = 0;
		SepalWidth = 0;
		PetalLength = 0;
		PetalWidth = 0;
		Species = "";
		IsSetosa = false;
	}

	// Constructor
	Iris(float data[NUM_FEATURES]) {
		SepalLength = data[0];
		SepalWidth = data[1];
		PetalLength = data[2];
		PetalWidth = data[3];
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

	void Print() {
		printf("Sepal Length: %f\n", SepalLength);
		printf("Sepal Width: %f\n", SepalWidth);
		printf("Petal Length: %f\n", PetalLength);
		printf("Petal Width: %f\n", PetalWidth);
		printf("Species: %s\n", Species);
	}
};
