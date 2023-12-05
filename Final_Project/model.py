import torch.nn as nn
import torch.nn.functional as F

class ExoplanetModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, output_size):
        """
        Initializes the ExoplanetModel class using hidden_size1, hidden_size2, hidden_size3, hidden_size4, and hidden_size5 as the sizes of the hidden layers.

        Args:
            input_size (int): The size of the input layer.
            hidden_size1 (int): The size of the first hidden layer.
            hidden_size2 (int): The size of the second hidden layer.
            hidden_size3 (int): The size of the third hidden layer.
            hidden_size4 (int): The size of the fourth hidden layer.
            hidden_size5 (int): The size of the fifth hidden layer.
            output_size (int): The size of the output layer.
        """
        super(ExoplanetModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.fc5 = nn.Linear(hidden_size4, hidden_size5)
        self.fc6 = nn.Linear(hidden_size5, output_size)
        
    def forward(self, x):
        """
        self.forward: produces the forward pass of the neural network model as an output tensor for the model.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        x = F.relu(self.fc1(x))  # Apply Leaky ReLU activation to the first layer
        x = F.relu(self.fc2(x))  # Apply Leaky ReLU activation to the second layer
        x = F.relu(self.fc3(x))  # Apply Leaky ReLU activation to the third layer
        x = F.relu(self.fc4(x))  # Apply Leaky ReLU activation to the fourth layer
        x = F.relu(self.fc5(x))  # Apply Leaky ReLU activation to the fifth layer
        x = self.fc6(x)                # No activation function on the output layer
        return x
    

# class ExoplanetModel(nn.Module):
#     def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
#         super(ExoplanetModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         # self.fc3 = nn.Linear(hidden_size2, hidden_size3)
#         self.fc4 = nn.Linear(hidden_size2, output_size)
        
#     def forward(self, x):
#         x = F.sigmoid(self.fc1(x))  # Apply ReLU activation to the first layer
#         x = F.sigmoid(self.fc2(x))  # Apply ReLU activation to the second layer
#         x = F.sigmoid(self.fc4(x))          # No activation function on the output layer
#         return x