#include <fstream>
#include "file_io.h"
#include "data_manip.h"
#include "neural_net.h"
int main(void)
{   
    std::string filename = "iris.csv";

    int height = 150, width = 5;
    int numbytes = height * width * sizeof(float);
    float *dataset = (float *) malloc(numbytes);
    if(!read_data(dataset, filename, width, false))
    {
        printf("Error reading data\n");
        return -1;
    }
    printf("File read successfully\n");
    
    int train_size = 130;

    float *train = (float *) malloc(train_size * width * sizeof(float));
    float *test = (float *) malloc((height - train_size) * width * sizeof(float));
    split_data(dataset, height, width, train, test, train_size);
    // validate_split(train, test, height, width, train_size);

    

    // int topology[] = {5, 3, 3};

    // NeuralNetwork net(topology, 3);
    printf("\n%f\n", dataset[0]);
    // float *inputs, *weights, *outputs;
    // int a = 2, b = 3,c = 4;
    // inputs = (float *) malloc(a * b * sizeof(float));
    // weights = (float *) malloc(b * c * sizeof(float));
    // outputs = (float *) malloc(a * c * sizeof(float));

    // for(int i = 0; i < a * b; i++)
    // {
    //     inputs[i] = i;
    // }
    // for(int i = 0; i < b * c; i++)
    // {
    //     weights[i] = i;
    // }

    // matrix_multiplication(inputs,weights,outputs,a,b,c);

    // for (int x = 0; x < a; x++) {
    //     for (int y = 0; y < c; y++) {
    //         printf("%f, ", outputs[x * c + y]);
    //     }
    //     printf("\n");
    // }
    // return 0;
}