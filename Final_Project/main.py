import os
import numpy as np
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ExoplanetModel
from data.dataset import ExoplanetDataset
from torch.utils.data import DataLoader, Dataset

def calculate_accuracy(output, label, percentage=5):
    """
    calculate_accuracy: produces the accuracy of the model's predictions.

    Parameters:
    output (torch.Tensor): The predicted output of the model.
    label (torch.Tensor): The true label values.
    percentage (float, optional): The tolerance percentage for considering a prediction as accurate. Default is 5.

    Returns:
    float: The accuracy of the model's predictions.
    """
    diff = torch.abs(output - label) # Calculate the absolute difference between the predicted and true values
    within_tolerance = (diff / label) <= percentage / 100.0 # Calculate the percentage difference between the predicted and true values
    accuracy = torch.mean(within_tolerance.float()).item() # Calculate the mean of the within_tolerance tensor
    return accuracy # Return the accuracy

def train(input_size, hidden_size, output_size, learning_rate, train_loader, epochs):
    """
    train: trains the neural network model using the given parameters for the ExoplanetModel and returns
    the trained model, a list of training losses, and a list of training accuracies.

    Args:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden layers.
        output_size (int): The size of the output.
        learning_rate (float): The learning rate for the optimizer.
        train_loader (DataLoader): The data loader for training data.
        epochs (int): The number of epochs to train the model.

    Returns:
        tuple: A tuple containing the trained model, a list of training losses, and a list of training accuracies.
    """
    # Create directory if it doesn't exist
    if not os.path.exists('result_dir'):
        os.makedirs('result_dir')

    # Initialize hidden layer sizes
    hidden_size1 = hidden_size
    hidden_size2 = hidden_size
    hidden_size3 = hidden_size
    hidden_size4 = hidden_size
    hidden_size5 = hidden_size

    # Initialize the model, optimizer, and loss function
    model = ExoplanetModel(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, output_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    loss_list = []
    accuracy_list = []
    for epoch in range(epochs):
        running_accuracy = 0.0
        train_loss = 0.0
        # Set model to training mode
        model.train()
        
        # Iterate over data in batches
        for data in train_loader:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Pass the batch of features to the model
            output = model(data['features'])
            
            # Calculate the loss
            loss = criterion(output, data['radius'])
            
            # Backpropagate the loss
            loss.backward()
            
            # Update the weights
            optimizer.step()

            running_accuracy += calculate_accuracy(output, data['radius'])
            train_loss += loss.item()

        accuracy = running_accuracy / len(train_loader) # Calculate the accuracy for the epoch
        loss_list.append(train_loss / len(train_loader)) # Calculate the loss for the epoch
        accuracy_list.append(accuracy) # Append the accuracy to the list of accuracies
            
        # Calculate loss and accuracy for the epoch
        print(f'Training Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}, Accuracy: {accuracy}')
        
        
    return model, loss_list, accuracy_list # Return the trained model, list of losses, and list of accuracies


def main():
    """
    The main function call to trains the neural network model to predict the radii of exoplanets.

    This function parses command line arguments, loads hyperparameters from the JSON file,
    creates directories for storing results, prepares the dataset and data loaders,
    trains the model, evaluates the model on the test set, and saves the results.

    Args:
        None

    Returns:
        None
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script trains a neural network model to predict the radii of exoplanets.')
    parser.add_argument('--param', type=str, default='./param/param.json', help='path to file containing hyperparamers: param/param.json')
    args = parser.parse_args()

    config_file_path = args.param # Path to the JSON file containing the hyperparameters

    # Load hyperparameters from the JSON file
    with open(config_file_path, 'r') as config_file:
        hyperparameters = json.load(config_file)

    # Initialize hyperparameters
    learning_rate = hyperparameters["learning_rate"]
    batch_size = hyperparameters["batch_size"]
    num_epochs = hyperparameters["epochs"]
    hidden_size = hyperparameters["hidden_size"]

    # Create 'result_dir' directory if it doesn't exist
    if not os.path.exists('result_dir'):
        os.makedirs('result_dir')

    # Create directory for the hyperparameters if it doesn't exist
    result_dir = f'result_dir/lr:{learning_rate}_bs:{batch_size}_epochs:{num_epochs}_hs:{hidden_size}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    input_size = 7 # Number of feature parameters per exoplanet radii prediction
    output_size = 1 # Number of output parameters per exoplanet radii prediction
    train_dataset = ExoplanetDataset('./data/training.csv') # Load the training dataset
    test_dataset = ExoplanetDataset('./data/testing.csv') # Load the testing dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Create a data loader for the training dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) # Create a data loader for the testing dataset
    trained_model, loss_list, accuracy_list = train(input_size, hidden_size, output_size, learning_rate, train_loader, num_epochs) # Train the model

    # Evaluate the model on the test set
    trained_model.eval() # Set the model to evaluation mode
    test_running_accuracy = 0.0 # Initialize the running accuracy
    test_loss = 0.0 # Initialize the test loss
    predicted_values = [] # Initialize the list of predicted values
    actual_values = [] # Initialize the list of actual values
    with torch.no_grad(): # Disable gradient calculation
        for data in test_loader: # Iterate over data in batches
            output = trained_model(data['features']) # Pass the batch of features to the model
            test_loss += nn.MSELoss()(output, data['radius']) # Calculate the test loss using MSE loss
            test_running_accuracy += calculate_accuracy(output, data['radius']) # Calculate the test accuracy
            predicted_values.extend(output)
            actual_values.extend(data['radius'])
    test_accuracy = test_running_accuracy / len(test_loader) # Calculate the test accuracy
    test_loss = test_loss / len(test_loader) # Calculate the test loss
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}') # Print the test loss and test accuracy
    
     # Plot the training loss over epochs
    plt.plot(range(1, num_epochs + 1), loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.savefig(f'{result_dir}/training_loss.png')  # Save the plot as an image file
    plt.show()

    # Plot the training accuracy over epochs
    plt.plot(range(1, num_epochs + 1), accuracy_list)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.savefig(f'{result_dir}/training_accuracy.png')  # Save the plot as an image file
    plt.show()

    # Create an array representing the number of items in the test loader
    num_items = np.arange(len(predicted_values))

    # Plot the graph for predicted values vs measured values
    plt.plot(num_items, predicted_values, label='Predicted Values')
    plt.plot(num_items, actual_values, label='Measured Values')
    plt.xlabel('Number of Items')
    plt.ylabel('Values')
    plt.title('Predicted Values vs Measured Values')
    plt.legend()
    plt.savefig(f'{result_dir}/predicted_vs_measured.png')  # Save the plot as an image file
    plt.show()

    predicted_values = np.array(predicted_values) # Convert the predicted values to a NumPy array
    actual_values = np.array(actual_values) # Convert the actual values to a NumPy array

    # Save the predicted and actual values in a text file
    np.savetxt(f'{result_dir}/predicted_vs_measured.txt', np.column_stack((predicted_values, actual_values)), delimiter='\t', header='Predicted\tMeasured', comments='')

# Run the main function
if __name__ == '__main__':
    main()