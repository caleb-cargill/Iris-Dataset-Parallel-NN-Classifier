#ifndef FILE_IO_H
#define FILE_IO_H

#include <fstream>
#include <string>
#include <sstream>
#include "matrix.h"


bool file_exists (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

bool read_data(struct Matrix *dataset, std::string filename, bool verbose)
{
    if (!file_exists(filename))
    {
        printf("File %s does not exist\n", filename.c_str());
        return false;
    }
    std::string line, word;
    std::ifstream infile;
    infile.open(filename);
    int i = 0;
    
    while (std::getline(infile, line))
    {   
        if (verbose) printf("\nRow #%i\n", i + 1);
        
        std::stringstream s(line);
        int j = 0;
        while (getline(s, word, ',')) {
			dataset->data[i*dataset->width+j] = std::stof(word);

			// print for validation
			if (verbose) printf("%i: %s, ", j, word);
			j++;
		}
        i++;
    }
    return true;
}

#endif