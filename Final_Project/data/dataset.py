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
    

class ExoplanetDataset2(Dataset):
    def __init__(self, x_df, y_df, preprocess=True, mode='train'):
        """
        Custom dataset example for PyTorch that accepts pandas DataFrame
        Args:
            x_df (DataFrame): A DataFrame containing the features (input data).
            y_df (DataFrame): A DataFrame containing the labels.
            mode (str): 'train' if the dataset is for training, 'test' for testing.
        """
        self.x_data = torch.tensor(x_df.values, dtype=torch.float32)
        self.y_data = torch.tensor(y_df.values, dtype=torch.float32)
        self.mode = mode
        self.preprocess = preprocess
        if preprocess:
            # Normalize features
            x_mean = self.x_data.mean(axis=0)
            x_std = self.x_data.std(axis=0)
            self.x_data = (self.x_data - x_mean) / x_std
            # limit y range
            y_range = self.y_data.max() -  self.y_data.min()
            self.y_data = (self.y_data - self.y_data.min()) / y_range

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.x_data)

    def __getitem__(self, index):
        """
        Generate one sample of data.
        """
        return {
            'features': self.x_data[index],
            'radius': self.y_data[index]
        }