## Iris Dataset Parallel NN Classifier

#### Team Members
 - Ben Elfner (elfnerbm@mail.uc.edu)
 - Caleb Cargill (cargilch@mail.uc.edu)

#### Background and Motivation
We have always been fascinated by machine learning and especially neural networks
and have worked with programs that abstract their implementations. It would be
interesting to take a closer look to see how neural networks and their training through
backpropagation can be optimized with the GPU. If possible, we would also like to test
changes to the architecture of the neural network and observe how it affects its
effectiveness. A potential advisor for this project is Professor Anca Ralescu
(anca.ralescu@uc.edu), a professor at the University of Cincinnati who teaches CS
5137, Machine Learning.

More information on the Iris Dataset can be found here: https://en.wikipedia.org/wiki/Iris_flower_data_set

#### Application-Level Objectives
Must-Have Features: 
 - An implementation of a feed-forward neural network that takes advantage of the GPU
 - An implementation of backpropagation that takes advantage of the GPU
 - The ability to train a classification neural network on the Iris Dataset
 - The trained neural network can classify and store the results for entries of the Iris Dataset

Optional Features: 
 - The ability to pass a filename to the program on the command line
 - The ability to train the neural network using other datasets
 - The ability to use different normalization functions


#### Performance Goals and Validation
In order to validate performance on this parallel classifier, we will write a benchmark
program that does not use parallelization to train on the dataset and classify items. With
a base program like this, we can track accuracy and program execution time and
compare it to the parallelized version of the classifier. We should be able to run both
programs, and compare accuracies and execution time. Ideally, we find that the
parallelized version of the classifier is at least as accurate as the standard classifier and
that the execution time is also less on the parallelized version than on the standard
classifier.

#### Schedule and Division of Work

Schedule:

| Task                                                              | Target Date   |
| :---                                                              | ---:          |
| Basic project structure done, work started on NN implementation   | 11/07/2021    |
| NN implementation completed, work started on backpropagation      | 11/14/2021    |
| Backpropagation implementation completed                          | 11/25/2021    |
| NN training on Iris Dataset working                               | 11/28/2021    |
| Presentation and all other deliverables completed                 | 11/30/2021    |

Division of Work: 

| Work                      | Ben       | Caleb     |
| :---                      | :----:    | ---:      |
| Neural Network            | 50%       | 50%       |
| Backpropagation           | 50%       | 50%       |
| All Other Code            | 50%       | 50%       |
| Final Deliverables        | 50%       | 50%       |
