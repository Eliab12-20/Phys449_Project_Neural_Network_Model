# PHYS449 - Final Project

## Dependencies
Please ensure that the following are installed/maintained to run the code for the Final Project:

- sys & os
- python
- json
- numpy
- argparse
- torch (Pytorch)
    - torch.optim
    - torch.nn
    - torch.utils.data
- torchvision 
    - torchvision.utils
- matplotlib.pyplot
- pandas

## Description of Task and Solution

Note that CLI means 'Command Line Interface'.

### Description of Task
For our final project we explored the paper by Ulmer-Moll et al. in 2019, 'Beyond the exoplanet mass-radius relation', and aimed to generate the radii predicitions for exoplanets and determine the accuracy of our predictions; specifically with a neural network to conduct these predictions. Our plans included implementing and training our model, calculating the prediction radius and its error, and visualzing these results. 

### Solution
To implement the solution for this final project, we first developed a dataset class to initialize the Exoplanet dataset as stored in our 'data' directory as a .csv file. We split up the data as testing and training data to ensure we can use this for our testing of the model. From this step, the neural network model was generated as ExoplanetModel

For the training loop, we calculated the accuracy of the model's prediction for the exoplanet radii, and trained the neural network based on the hyperparameters of: number of epochs, batch size, hidden size, and learning rate. We calculated the MSELoss to identify the loss for the model, and used the Adam optimizer.

Finally, we generated four main output files for each model run:
- An graph showing predicted vs measured exoplanet radii
- The values stored as a .txt file for these predicted vs measured values
- The training accuracy as a function of the number of epochs
- The training loss as a function of epochs

In this way, we were able to provide quantitative evidence as to our model's performance, and comparison to the literature. For more information on our choice of hyperparameters, see below.

## AI Statement
For this assignment, we made use of ChatGPT and Github Co-pilot. ChatGPT was used to debug any issues in the CLI interface when running the code, and helped to setup the initial model and data processing scripts. Co-pilot was helpful to aid in the creation of documentation and docstrings, which were edited to better reflect the use of the code.

## Running `main.py`
To run `main.py`, first clone repository on personal machine. Do this by using the command line to go to a directory to clone: https://github.com/yasairam/PHYS449.git. Then, use the change directory command to go to 'PHYS449_PROJECT_NEURAL_NETWORK_MODEL'. Then, use the change directory command once in that directory to go to 'Final_Project'. 

To run `main.py`, use the following as an example:

```sh
python main.py --param ./param/param.json 
```

Output on the CLI is provided as below, given the --verbose flag is used (Loss values may differ for different hyperparameters):
```
Training Epoch [1/500], Loss: 58.06696319580078, Accuracy: 0.010416666666666666
Training Epoch [2/500], Loss: 9.897356986999512, Accuracy: 0.10364583383003871
Training Epoch [3/500], Loss: 5.065518379211426, Accuracy: 0.14895833345750967
Training Epoch [4/500], Loss: 14.264244079589844, Accuracy: 0.18125000006208816
Training Epoch [5/500], Loss: 9.992265701293945, Accuracy: 0.169791666790843
Training Epoch [6/500], Loss: 9.409388542175293, Accuracy: 0.154166666790843
Training Epoch [7/500], Loss: 23.511363983154297, Accuracy: 0.20729166672875485
Training Epoch [8/500], Loss: 11.762777328491211, Accuracy: 0.17604166672875485
Training Epoch [9/500], Loss: 23.66176986694336, Accuracy: 0.18385416672875485
Training Epoch [10/500], Loss: 12.25613021850586, Accuracy: 0.1942708333954215
Training Epoch [11/500], Loss: 16.857328414916992, Accuracy: 0.22343750049670538
Training Epoch [12/500], Loss: 1.9288793802261353, Accuracy: 0.19114583358168602
Training Epoch [13/500], Loss: 8.462017059326172, Accuracy: 0.20052083333333334
Training Epoch [14/500], Loss: 2.955626964569092, Accuracy: 0.185416666790843
Training Epoch [15/500], Loss: 8.784649848937988, Accuracy: 0.19166666672875485
... 
Test Loss: 38.71796417236328, Test Accuracy: 0.16517857182770967
```

Where the iterative report for the loss value and accuracy is provided through the training epochs. At the end of all epochs ran, the Test Loss and Test accuracy is written out.

## Hyperparameters (stored in param.json)
From the param.json file, we see there are parameters such as: learning rate, batch size, hidden size and epochs. 

### param.json example
```sh
{
    "learning_rate": 0.01,
    "batch_size": 16,
    "hidden_size": 10,
    "epochs": 500
}
```

These values were determined to optimize the training model based on my testing. We found that setting the learning rate specifically to 0.1 was optimal to generate the lowest loss for this model. We also tried to ensure it was not too low to lead to overfitting, and shared this sentiment for the number of epochs with this in mind.


## Help Command

To get help for `main.py`, run the following command on the CLI:

```sh
python main.py --help
```

The output for running the --help command is given below:

```sh
usage: main.py [-h] [--param PARAM]

This script trains a neural network model to predict the radii of exoplanets.

options:
  -h, --help     show this help message and exit
  --param PARAM  path to file containing hyperparamers: param/param.json
```