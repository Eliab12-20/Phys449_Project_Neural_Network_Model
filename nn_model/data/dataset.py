import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class ExoplanetDataset(Dataset):
    def __init__(self, csv_file):
        # Load the dataset
        self.data = pd.read_csv(csv_file, skiprows=[0])
        
        # Drop rows with missing values if any
        self.data.dropna(inplace=True)
        
        # Separate exoplanet names and measured radii
        self.exoplanet_names = self.data.iloc[:, 0].values.tolist()
        self.measured_radii = self.data.iloc[:, -1].values.reshape(-1, 1).astype('float32')
        
        # Extract features (excluding exoplanet name) for training
        self.features = self.data.iloc[:, 1:-1].values.astype('float32')
        
        # Normalize features
        self.feature_means = self.features.mean(axis=0)
        self.feature_std = self.features.std(axis=0)
        self.features = (self.features - self.feature_means) / self.feature_std
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx]),
            'radius': torch.tensor(self.measured_radii[idx]),
            'exoplanet_name': self.exoplanet_names[idx]
        }