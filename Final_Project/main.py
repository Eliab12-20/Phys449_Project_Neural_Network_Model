import os
import numpy as np
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ExoplanetModel
from data.dataset import ExoplanetDataset, ExoplanetDataset2
from torch.utils.data import DataLoader, Dataset
import data.load_data as ld
from sklearn.model_selection import train_test_split

exoplanet_path = 'data/exoplanet.eu_catalog_15April.csv'
solar_path = 'data/solar_system_planets_catalog.csv'
features_name = ['mass', 'semi_major_axis','eccentricity', 'star_metallicity',
                'star_radius', 'star_teff','star_mass', 'radius']


def calculate_error(output, label, percentage=5):
    """
    calculate_error: produces the error of the model's predictions.

    Parameters:
    output (torch.Tensor): The predicted output of the model.
    label (torch.Tensor): The true label values.
    percentage (float, optional): The tolerance percentage for considering a prediction as accurate. Default is 5.

    Returns:
    float: The error of the model's predictions.
    """
    diff = torch.abs(output - label) # Calculate the absolute difference between the predicted and true values
    within_tolerance = (diff / label) <= percentage / 100.0 # Calculate the percentage difference between the predicted and true values
    error = torch.mean(within_tolerance.float()).item() # Calculate the mean of the within_tolerance tensor
    return error # Return the error

def calculate_err(pred, label):
    err = torch.abs(pred - label) / label
    err_array = err.detach().numpy()
    err_array = err_array[np.isfinite(err_array)]
    err_mean = err_array.mean()
    return err_mean

def train(input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, hidden_size_5, output_size, learning_rate, train_loader, epochs):
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
    hidden_size1 = hidden_size_1
    hidden_size2 = hidden_size_2
    hidden_size3 = hidden_size_3
    hidden_size4 = hidden_size_4
    hidden_size5 = hidden_size_5

    # Initialize the model, optimizer, and loss function
    model = ExoplanetModel(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, output_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    loss_list = []
    error_list = []
    for epoch in range(epochs):
        running_error = 0.0
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

            running_error += calculate_err(output, data['radius'])
            train_loss += loss.item()

        error = running_error / len(train_loader) # Calculate the error for the epoch
        loss_list.append(train_loss / len(train_loader)) # Calculate the loss for the epoch
        error_list.append(error) # Append the error to the list of accuracies
            
        # Calculate loss and error for the epoch
        print(f'Training Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}, Error: {error}')
        
        
    return model, loss_list, error_list # Return the trained model, list of losses, and list of accuracies


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
    hidden_size1 = hyperparameters["hidden_size1"]
    hidden_size2 = hyperparameters["hidden_size2"]
    hidden_size3 = hyperparameters["hidden_size3"]
    hidden_size4 = hyperparameters["hidden_size4"]
    hidden_size5 = hyperparameters["hidden_size5"]

    # Create 'result_dir' directory if it doesn't exist
    if not os.path.exists('result_dir'):
        os.makedirs('result_dir')

    # Create directory for the hyperparameters if it doesn't exist
    result_dir = f'result_dir/lr_{learning_rate}_bs_{batch_size}_epochs_{num_epochs}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    input_size = 7 # Number of feature parameters per exoplanet radii prediction
    output_size = 1 # Number of output parameters per exoplanet radii prediction
    train_dataset = ExoplanetDataset('./data/training.csv') # Load the training dataset
    test_dataset = ExoplanetDataset('./data/testing.csv') # Load the testing dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Create a data loader for the training dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) # Create a data loader for the testing dataset
    print("Training First Model With 7 Features")
    trained_model, loss_list, error_list = train(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, output_size, learning_rate, train_loader, num_epochs) # Train the model


    # Data from Dataset2 with 9 features


    # Load from 2nd Dataset with 9 features
    dataset2 = ld.load_dataset(exoplanet_path, solar_path, features_name, solar=True, add_feature=False)
    dataset2_exo = dataset2[:-8]
    #dataset2_solar = dataset2[-8:]
    features_2 = dataset2_exo.drop('radius', axis=1).copy()   # mass, teq, etc
    labels_2 = dataset2_exo[['radius']].copy()   # radius
    X_train, X_test, y_train, y_test = train_test_split(features_2, labels_2['radius'], test_size=0.25, random_state=0)
    X_test = X_test.drop(['HATS-12 b'])
    y_test = y_test.drop(labels=['HATS-12 b'])
    print('\nHATS-12 b removes from test set\n')

    # Remove K2-95 b from the training set
    X_train = X_train.drop(['K2-95 b'])
    y_train = y_train.drop(labels=['K2-95 b'])
    print('\nK2-95 b removes from training set\n')

    # Remove Kepler-11 g from the training set
    X_train = X_train.drop(['Kepler-11 g'])
    y_train = y_train.drop(labels=['Kepler-11 g'])
    print('\nKepler-11 g removes from training set\n')

    train_dataset_2 = ExoplanetDataset2(X_train, y_train, preprocess=True, mode='train')
    test_dataset_2 = ExoplanetDataset2(X_test, y_test, preprocess=True, mode='test')

    train_loader_2 = DataLoader(train_dataset_2, batch_size=len(train_dataset_2), shuffle=True)
    test_loader_2 = DataLoader(test_dataset_2, batch_size=len(test_dataset_2), shuffle=True)

    print("Training Second Model with 9 features")
    trained_model2, loss_list2, error_list2 = train(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, output_size, learning_rate, train_loader_2, num_epochs) # Train the model

    # input_size = dataset_exo.shape[-1] - 1
    # hidden_size_1 = 64
    # hidden_size_2 = 16
    # hidden_size_3 = 16
    # output_size = 1
    # learning_rate = 0.001
    # model = ExoplanetModel(input_size, hidden_size_1, hidden_size2, hidden_size3, output_size)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.MSELoss()



    # Evaluate the first model on the test set
    print("Evaluating First Model")
    trained_model.eval() # Set the model to evaluation mode
    test_running_error = 0.0 # Initialize the running error
    test_loss = 0.0 # Initialize the test loss
    predicted_values = [] # Initialize the list of predicted values
    actual_values = [] # Initialize the list of actual values
    with torch.no_grad(): # Disable gradient calculation
        for data in test_loader: # Iterate over data in batches
            output = trained_model(data['features']) # Pass the batch of features to the model
            test_loss += nn.MSELoss()(output, data['radius']) # Calculate the test loss using MSE loss
            test_running_error += calculate_err(output, data['radius']) # Calculate the test error
            predicted_values.extend(output)
            actual_values.extend(data['radius'])
    test_error = test_running_error / len(test_loader) # Calculate the test error
    test_loss = test_loss / len(test_loader) # Calculate the test loss
    print(f'Test Loss: {test_loss}, Test Error: {test_error}') # Print the test loss and test error
    

    # Evaluate the second model on the test set
    print("Evaluating Second Model")
    trained_model2.eval() # Set the model to evaluation mode
    test_running_error2 = 0.0 # Initialize the running error
    test_loss2 = 0.0 # Initialize the test loss
    predicted_values2 = [] # Initialize the list of predicted values
    actual_values2 = [] # Initialize the list of actual values
    with torch.no_grad(): # Disable gradient calculation
        for data in test_loader_2: # Iterate over data in batches
            output = trained_model2(data['features']) # Pass the batch of features to the model
            test_loss2 += nn.MSELoss()(output, data['radius']) # Calculate the test loss using MSE loss
            test_running_error2 += calculate_err(output, data['radius']) # Calculate the test error
            predicted_values2.extend(output)
            actual_values2.extend(data['radius'])
    test_error2 = test_running_error2 / len(test_loader_2) # Calculate the test error
    test_loss2 = test_loss2 / len(test_loader_2) # Calculate the test loss
    print(f'Test Loss: {test_loss2}, Test Error: {test_error2}') # Print the test loss and test error


     # Plot the training loss over epochs for Model 1
    plt.plot(range(1, num_epochs + 1), loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs For Model 1')
    plt.savefig(f'{result_dir}/training_loss.png')  # Save the plot as an image file
    plt.show()

    # Plot the training error over epochs for Model 1
    plt.plot(range(1, num_epochs + 1), error_list)
    plt.xlabel('Epoch')
    plt.ylabel('Training error')
    plt.title('Training error over Epochs For Model 1')
    plt.savefig(f'{result_dir}/training_error.png')  # Save the plot as an image file
    plt.show()

    # # Create an array representing the number of items in the test loader
    # num_items = np.arange(len(predicted_values))

    # # Plot the graph for predicted values vs measured values
    # plt.plot(num_items, predicted_values, label='Predicted Values')
    # plt.plot(num_items, actual_values, label='Measured Values')
    # plt.xlabel('Number of Items')
    # plt.ylabel('Values')
    # plt.title('Predicted Values vs Measured Values')
    # plt.legend()
    # plt.savefig(f'{result_dir}/predicted_vs_measured.png')  # Save the plot as an image file
    # plt.show()

    # predicted_values = np.array(predicted_values) # Convert the predicted values to a NumPy array
    # actual_values = np.array(actual_values) # Convert the actual values to a NumPy array

## Plot the relationship between predicted and measured values
    
    # Convert the predicted and actual values to 1D arrays
    predicted_values = np.ravel(np.array(predicted_values))
    actual_values = np.ravel(np.array(actual_values))
 
    # Calculate the coefficients and intercept of the linear regression line
    coefficients = np.polyfit(actual_values, predicted_values, 1)
 
    # Calculate the predicted values using the coefficients and intercept
    predicted_values_fit = coefficients[0] * actual_values + coefficients[1]
 
    # Calculate the r^2 score
    r2 = 1 - (np.sum((predicted_values - predicted_values_fit) ** 2) / ((len(predicted_values) - 1) * np.var(predicted_values, ddof=1)))
 
    # Create the equation of the line
    line_eq = f'y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}'
 
    # Plot the original scatter plot
    plt.figure()
    plt.scatter(actual_values, predicted_values, color='blue', s=15)
 
    # Plot the regression line
    plt.plot(actual_values, predicted_values_fit, color='green', label='Regression Line')
 
    # Add the equation of the line and the r^2 score as text on the plot
    plt.text(0.05, 0.95, f'{line_eq}, $R^2 = {r2:.2f}$', transform=plt.gca().transAxes)
 
    # Add the rest of the plot details
    plt.plot(actual_values, predicted_values, color='red', label='y=x')
    plt.xlabel('Measured Values ($\mathrm{R}_{\oplus}$)')
    plt.ylabel('Predicted Values ($\mathrm{R}_{\oplus}$)')
    plt.title('Model 1: Predicted Values vs. Measured Values ($\mathrm{R}_{\oplus}$)')
 
    # Move the legend to the right of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
 
    plt.axis('equal')  # Ensure that the axes are on the same scale
 
    # Adjust the layout to make room for the legend
    plt.tight_layout()
    
    plt.savefig(f'{result_dir}/Model_1_predicted_measured_relationship.png')  # Save the plot as an image file
    plt.show()
    
    # Save the predicted and actual values in a text file
    np.savetxt(f'{result_dir}/predicted_vs_measured.txt', np.column_stack((predicted_values, actual_values)), delimiter='\t', header='Predicted\tMeasured', comments='')



# Plot Graphs for Model 2

  # Plot the training loss over epochs for Model 1
    plt.plot(range(1, num_epochs + 1), loss_list2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs For Model 2')
    plt.savefig(f'{result_dir}/training_loss2.png')  # Save the plot as an image file
    plt.show()

    # Plot the training error over epochs for Model 1
    plt.plot(range(1, num_epochs + 1), error_list2)
    plt.xlabel('Epoch')
    plt.ylabel('Training error')
    plt.title('Training error over Epochs For Model 2')
    plt.savefig(f'{result_dir}/training_error.png')  # Save the plot as an image file
    plt.show()

    # # Create an array representing the number of items in the test loader
    # num_items = np.arange(len(predicted_values))

    # # Plot the graph for predicted values vs measured values
    # plt.plot(num_items, predicted_values, label='Predicted Values')
    # plt.plot(num_items, actual_values, label='Measured Values')
    # plt.xlabel('Number of Items')
    # plt.ylabel('Values')
    # plt.title('Predicted Values vs Measured Values')
    # plt.legend()
    # plt.savefig(f'{result_dir}/predicted_vs_measured.png')  # Save the plot as an image file
    # plt.show()

    # predicted_values = np.array(predicted_values) # Convert the predicted values to a NumPy array
    # actual_values = np.array(actual_values) # Convert the actual values to a NumPy array

## Plot the relationship between predicted and measured values
    
    # Convert the predicted and actual values to 1D arrays
    predicted_values2 = np.ravel(np.array(predicted_values2))
    actual_values2 = np.ravel(np.array(actual_values2))
 
    # Calculate the coefficients and intercept of the linear regression line
    coefficients = np.polyfit(actual_values2, predicted_values2, 1)
 
    # Calculate the predicted values using the coefficients and intercept
    predicted_values_fit = coefficients[0] * actual_values2 + coefficients[1]
 
    # Calculate the r^2 score
    r2 = 1 - (np.sum((predicted_values2 - predicted_values_fit) ** 2) / ((len(predicted_values2) - 1) * np.var(predicted_values2, ddof=1)))
 
    # Create the equation of the line
    line_eq = f'y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}'
 
    # Plot the original scatter plot
    plt.figure()
    plt.scatter(actual_values2, predicted_values2, color='blue', s=15)
 
    # Plot the regression line
    plt.plot(actual_values2, predicted_values_fit, color='green', label='Regression Line')
 
    # Add the equation of the line and the r^2 score as text on the plot
    plt.text(0.05, 0.95, f'{line_eq}, $R^2 = {r2:.2f}$', transform=plt.gca().transAxes)
 
    # Add the rest of the plot details
    plt.plot(actual_values2, predicted_values2, color='red', label='y=x')
    plt.xlabel('Measured Values ($\mathrm{R}_{\oplus}$)')
    plt.ylabel('Predicted Values ($\mathrm{R}_{\oplus}$)')
    plt.title('Model 2: Predicted Values vs. Measured Values ($\mathrm{R}_{\oplus}$)')
 
    # Move the legend to the right of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
 
    plt.axis('equal')  # Ensure that the axes are on the same scale
 
    # Adjust the layout to make room for the legend
    plt.tight_layout()
    
    plt.savefig(f'{result_dir}/Model_2_predicted_measured_relationship.png')  # Save the plot as an image file
    plt.show()
    
    # Save the predicted and actual values in a text file
    np.savetxt(f'{result_dir}/predicted_vs_measured2.txt', np.column_stack((predicted_values2, actual_values2)), delimiter='\t', header='Predicted\tMeasured', comments='')

# Run the main function
if __name__ == '__main__':
    main()
