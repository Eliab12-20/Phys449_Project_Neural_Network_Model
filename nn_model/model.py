import torch.nn as nn
import torch.nn.functional as F

class ExoplanetModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, output_size):
        super(ExoplanetModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.fc5 = nn.Linear(hidden_size4, hidden_size5)
        self.fc6 = nn.Linear(hidden_size5, output_size)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))  # Apply Leaky ReLU activation to the first layer
        x = F.leaky_relu(self.fc2(x))  # Apply Leaky ReLU activation to the second layer
        x = F.leaky_relu(self.fc3(x))  # Apply Leaky ReLU activation to the third layer
        x = F.leaky_relu(self.fc4(x))  # Apply Leaky ReLU activation to the fourth layer
        x = F.leaky_relu(self.fc5(x))  # Apply Leaky ReLU activation to the fifth layer
        x = self.fc6(x)                # No activation function on the output layer
        return x