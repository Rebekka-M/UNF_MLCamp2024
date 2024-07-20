"""Modul til definering af netværksarkitektur"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data._data_static import TEGNINGER
from options import Hyperparameters

C = len(TEGNINGER)

class Net(nn.Module):
    """
    Netværksarkitektur for klassifikation af billeder
    
    Args:
    nn.Module: Superklasse for alle neurale netværk i PyTorch
    
    Returns:
    Net: Netværksarkitektur
    """
    def __init__(self, name: str, hyperparameters: Hyperparameters):
        # Initialiserer architecturen
        super(Net, self).__init__()

        # Navngiv model
        self.name = name

        # Load Hyperparametre
        self.hyperparameters = hyperparameters

        # Vælg loss function
        self.criterion = nn.CrossEntropyLoss()
        setattr(self.hyperparameters, 'loss', self.criterion.__class__.__name__)

        # Definer lagene i netværket
        raise NotImplementedError("Implementer Netværksarkitektur. Det sidste lag er defineret nedenunder")

    def forward(self, x: torch.Tensor):
        """
        Forward pass af netværket
        
        Args:
        x (torch.Tensor): Input tensor
        
        Returns:
        torch.Tensor: Output tensor
        """
        raise NotImplementedError("Implementer forward pass")

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
    
    def save(self):
        """
        Gemmer modellen
        
        Args:
        path (str): Sti til gemmested
        """
        scripted_model = torch.jit.script(self)
        scripted_model.save(f'2.NN/QuickDraw/saved_models/{self.name}.pth')
