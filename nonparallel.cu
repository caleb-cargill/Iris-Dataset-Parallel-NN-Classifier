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

int main(void)
{
    std::ifstream infile("iris.csv");
    std::string line, word;
    int i = 0;
    float dataset[150][5];

    while (std::getline(infile, line))
    {
        printf("\nRow #%i\n", i + 1);
        std::stringstream s(line);
        int j = 0;
        while (getline(s, word, ',')) {
            dataset[i][j] = std::stof(word);

            // print for validation
            printf("%i: %s, ", j, word);
            j++;
        }
        i++;
    }
    printf("%f", dataset[0][0]);
    return 0;
}