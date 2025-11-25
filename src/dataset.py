#src/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd, numpy as np

class StockDataset(Dataset):
    def __init__(self, features:np.ndarray, target:np.ndarray):
        self.features = torch.tensor(features, dtype = torch.float)

        # Transfer the target variable and make sure the shape is (N, 1)
        self.target = torch.tensor(target, dtype = torch.float).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

