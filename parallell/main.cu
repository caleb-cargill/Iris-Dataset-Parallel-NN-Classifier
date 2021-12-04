#include <fstream>
#include "file_io.h"
#include "data_manip.h"
#include "neural_net.h"
#include "mat_mult.h"
#include "cuda.h"
#include "matrix.h"
#include <iostream>
#include "gputimer.h"

int main(void)
{   
    GpuTimer timer;
	timer.Start();

    std::string filename = "iris.csv";

    int height = 150, width = 5;
    struct Matrix *dataset = create_matrix(height, width);
    if(!read_data(dataset, filename, false))
    {
        printf("Error reading data\n");
        return -1;
    }
    normalize_data(dataset);

    int train_size = 100;

    struct Matrix *train, *test;
    train = create_matrix(train_size, width);
    test = create_matrix(height - train_size, width);
    split_data(dataset, train, test);

    struct Matrix *ground_truth = create_matrix(train_size, 3);
    create_ground_truth(train, ground_truth);

    int topology[] = {4, 3, 3};
    int epochs = 50;
    NeuralNetwork net(topology, 2);
    net.train(train,ground_truth,epochs,.001);
    printf("epochs: %d\n", epochs);
    timer.Stop();

	printf("\nTime Elapsed: %g ms\n", timer.Elapsed());
 
    return 0;
}