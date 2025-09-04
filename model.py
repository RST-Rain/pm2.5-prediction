import torch
import torch.nn as nn

# Custom Dataset for PyTorch
class AirQualityDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        """
        Initializes dataset with features (X) and target (y).
        Args:
            X: Numpy array of features.
            y: Numpy array of target (PM2.5 values).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Simple MLP Model
class AirPollutionModel(nn.Module):
    def __init__(self, input_dim):
        """
        Initializes MLP model for PM2.5 prediction.
        Args:
            input_dim: Number of input features.
        """
        super(AirPollutionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        """
        Defines forward pass of the model.
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        Returns:
            Output tensor of shape (batch_size, 1).
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x