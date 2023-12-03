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
    diff = torch.abs(output - label)
    within_tolerance = (diff / label) <= percentage / 100.0
    accuracy = torch.mean(within_tolerance.float()).item()
    return accuracy

def train(input_size, hidden_size, output_size, learning_rate, train_loader, epochs):
    # Create directory if it doesn't exist
    if not os.path.exists('result_dir'):
        os.makedirs('result_dir')

    hidden_size1 = hidden_size
    hidden_size2 = hidden_size
    hidden_size3 = hidden_size

    model = ExoplanetModel(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
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

        accuracy = running_accuracy / len(train_loader)
        loss_list.append(train_loss / len(train_loader))
        accuracy_list.append(accuracy)
            
        # Calculate loss and accuracy for the epoch
        print(f'Training Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}, Accuracy: {accuracy}')
        
        
    return model, loss_list, accuracy_list

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script trains a neural network model to predict the radii of exoplanets.')
    parser.add_argument('--param', type=str, default='./param/param.json', help='path to file containing hyperparamers: param/param.json')
    args = parser.parse_args()

    config_file_path = args.param

    with open(config_file_path, 'r') as config_file:
        hyperparameters = json.load(config_file)

    learning_rate = hyperparameters["learning_rate"]
    batch_size = hyperparameters["batch_size"]
    num_epochs = hyperparameters["epochs"]
    hidden_size = hyperparameters["hidden_size"]

    input_size = 7 # Number of feature parameters per exoplanet radii prediction
    output_size = 1
    train_dataset = ExoplanetDataset('./data/training.csv')
    test_dataset = ExoplanetDataset('./data/testing.csv')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    trained_model, loss_list, accuracy_list = train(input_size, hidden_size, output_size, learning_rate, train_loader, num_epochs)

    # Evaluate the model on the test set
    trained_model.eval()
    test_running_accuracy = 0.0
    test_loss = 0.0
    predicted_values = []
    actual_values = []
    with torch.no_grad():
        for data in test_loader:
            output = trained_model(data['features'])
            test_loss += nn.MSELoss()(output, data['radius'])
            test_running_accuracy += calculate_accuracy(output, data['radius'])
            predicted_values.extend(output)
            actual_values.extend(data['radius'])
    test_accuracy = test_running_accuracy / len(test_loader)
    test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
    
    # Plot the training loss over epochs
    plt.plot(range(1, num_epochs + 1), loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.savefig(f'result_dir/training_loss_lr:{learning_rate}_bs:{batch_size}_epochs:{num_epochs}_hs:{hidden_size}.png')  # Save the plot as an image file
    plt.show()

    # Plot the training accuracy over epochs
    plt.plot(range(1, num_epochs + 1), accuracy_list)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.savefig(f'result_dir/training_accuracy_lr:{learning_rate}_bs:{batch_size}_epochs:{num_epochs}_hs:{hidden_size}.png')  # Save the plot as an image file
    plt.show()

    # Create an array representing the number of items in the test loader
    num_items = np.arange(len(predicted_values))

    # Plot the graph
    plt.plot(num_items, predicted_values, label='Predicted Values')
    plt.plot(num_items, actual_values, label='Measured Values')
    plt.xlabel('Number of Items')
    plt.ylabel('Values')
    plt.title('Predicted Values vs Measured Values')
    plt.legend()
    plt.savefig(f'result_dir/predicted_vs_measured_lr:{learning_rate}_bs:{batch_size}_epochs:{num_epochs}_hs:{hidden_size}.png')  # Save the plot as an image file
    plt.show()

    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)

    # Save the predicted and actual values in a text file
    np.savetxt(f'result_dir/predicted_vs_measured_lr:{learning_rate}_bs:{batch_size}_epochs:{num_epochs}_hs:{hidden_size}.txt', np.column_stack((predicted_values, actual_values)), delimiter='\t', header='Predicted\tMeasured', comments='')

if __name__ == '__main__':
    main()