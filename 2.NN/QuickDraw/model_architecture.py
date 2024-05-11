import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class Net(nn.Module):
    """
    Netværksarkitektur for klassifikation af billeder
    
    Args:
    nn.Module: Superklasse for alle neurale netværk i PyTorch
    
    Returns:
    Net: Netværksarkitektur
    """
    def __init__(self, hyperparameters: dict = {}, C: int = 3):
        # Initialiserer architecturen
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, C)

        # Vælg loss function og optimizer
        self.criterion = nn.CrossEntropyLoss
        self.optimizer = optim.SGD
        self.hyperparameters = hyperparameters

    def forward(self, x: torch.Tensor):
        """
        Forward pass af netværket
        
        Args:
        x (torch.Tensor): Input tensor
        
        Returns:
        torch.Tensor: Output tensor
        """
        x = x.reshape(-1, 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    def predict(self, x: np.ndarray):
        """
        Forudsig klasse
        
        Args:
        x (np.ndarray): Input data
        
        Returns:
        torch.Tensor: Forudsiget klasse
        """
        # Konverter til tensor
        x = torch.Tensor(x).float()
        x = x.reshape(-1, 1, 28, 28)

        # Forudsig klasse
        y_hat_prob = self(x)
        y_hat = torch.argmax(y_hat_prob, dim=1)

        return y_hat.detach().numpy()[0], y_hat_prob[0].detach().numpy()