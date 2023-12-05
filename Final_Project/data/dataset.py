import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class ExoplanetDataset(Dataset):
    def __init__(self, csv_file):
        """
        Initializes the Exoplanet Dataset object.

        Parameters:
        - csv_file (str): The path to the CSV file containing the dataset.

        Returns:
        None
        """
        # Load the dataset, skipping the header row
        self.data = pd.read_csv(csv_file, skiprows=[0])
        
        # Drop rows with missing values if any
        self.data.dropna(inplace=True)
        
        # Extract features and measured radii
        self.features = self.data.iloc[:, 1:-1].values.astype('float32')
        self.measured_radii = self.data.iloc[:, -1].values.astype('float32').reshape(-1, 1)
        
        # Normalize features
        self.feature_means = self.features.mean(axis=0)
        self.feature_std = self.features.std(axis=0)
        self.features = (self.features - self.feature_means) / self.feature_std
        
        # Convert to PyTorch tensors
        self.features = torch.tensor(self.features)
        self.measured_radii = torch.tensor(self.measured_radii)
        
    def __len__(self):
        """
        self.__len__: returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        self.__getitem__: produces the item at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the features and radius of the item.
        """
        return {
            'features': self.features[idx],
            'radius': self.measured_radii[idx]
        }