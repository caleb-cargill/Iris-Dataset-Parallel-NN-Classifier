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

static const int DATASET_SIZE = 150;
static const int NUM_FEATURES = 5;
static Iris iris_dataset[DATASET_SIZE];

void init_dataset() {
    std::ifstream infile("iris.csv");
    std::string line, word;
    int i = 0;
    float raw_dataset[DATASET_SIZE][NUM_FEATURES];


    while (std::getline(infile, line))
    {
        printf("\nRow #%i\n", i + 1);
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

    // Print Dataset for Validation
    for (int i = 0; i < DATASET_SIZE; i++) {
        iris_dataset[i].Print();
    }

    return 0;
}