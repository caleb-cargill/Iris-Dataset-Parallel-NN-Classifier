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

static const int DATASET_SIZE = 150; // number of rows in the dataset
static const int NUM_FEATURES = 5; // number of features in the dataset; 4 input, 1 output
static Iris iris_dataset[DATASET_SIZE]; // variable to store dataset in
static const std::string DATASET_FILE_NAME = "iris.csv"; // file name that the dataset is stored in

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

int main(void)
{
    init_dataset();

    // Split Dataset into Training and Testing Data
    // 2/3 Training, 1/3 Testing
    Iris training_data[DATASET_SIZE * 2 / 3];
    Iris testing_data[DATASET_SIZE / 3];
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

    return 0;
}