#ifndef FILE_IO_H
#define FILE_IO_H

#include <fstream>
#include <string>
#include <sstream>

/// <summary>
/// Checks to see if a file named as the passed in filename exists
/// </summary>
/// <param name="name"></param>
/// <returns></returns>
bool file_exists (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

/// <summary>
/// Reads the data within the passed in filename and fills the dataset array with data values
/// </summary>
/// <param name="dataset"></param>
/// <param name="filename"></param>
/// <param name="width"></param>
/// <param name="verbose"></param>
/// <returns>Boolean denoting success or failure</returns>
bool read_data(float *dataset, std::string filename, int width, bool verbose)
{
    // Check to see if the file exists
    if (!file_exists(filename))
    {
        printf("File %s does not exist\n", filename.c_str());
        return false;
    }

    // Initialize reader variables
    std::string line, word;
    std::ifstream infile;
    infile.open(filename);
    int i = 0;
    
    // Read each line in the file
    while (std::getline(infile, line))
    {   
        if (verbose) printf("\nRow #%i\n", i + 1);
        
        std::stringstream s(line);
        int j = 0;

        // Read each value in the comma separated list
        while (getline(s, word, ',')) {
            // Convert value to float and place in dataset array
			dataset[i*width+j] = std::stof(word);

			// print for validation
			if (verbose) printf("%i: %s, ", j, word);
			j++;
		}
        i++;
    }

    return true;
}

#endif