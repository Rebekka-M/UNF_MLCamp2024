"""Modul til definering af netværksarkitektur"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data._data_static import TEGNINGER
from get_data import get_dataset
from options import Hyperparameters, name_generator
from train import train

import numpy as np

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

        input_size = 28
        num_classes = C
        #self.num_classes = num_classes
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 5, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        dimension = int(64 * pow(input_size/4 - 3, 2))
        self.fc1 = nn.Sequential(nn.Linear(dimension, 512), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(512, 128), nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(128, num_classes))

        # Definer lagene i netværket
        #raise NotImplementedError("Implementer Netværksarkitektur. Det sidste lag er defineret nedenunder")

    def forward(self, x: torch.Tensor):
        """
        Forward pass af netværket
        
        Args:
        x (torch.Tensor): Input tensor
        
        Returns:
        torch.Tensor: Output tensor
        """
        x = x.reshape(-1, 1, 28, 28)
        output = self.conv1(x)
        output = self.conv2(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        
        #raise NotImplementedError("Implementer forward pass")

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
    
#if __name__ == "__main__":

    # # Sæt valgmuligheder
    # hyperparameters = Hyperparameters(
    #     lr = 0.005,
    # )

    # # Hent data fra get_data.py
    # train_loader, val_loader = get_dataset(
    #     names=TEGNINGER,
    #     n_samples=1000,
    #     batch_size=hyperparameters.batch_size,
    #     verbose=True
    # )

    # # Hent model architecturene fra model_architecture.py
    # model = Net(
    #     name = name_generator(),
    #     hyperparameters=hyperparameters
    # )

    # # tilføj optimizer til model
    # model.optimizer = model.hyperparameters.optimizer(
    #     model.parameters(),
    #     lr=model.hyperparameters.lr,
    #     momentum=model.hyperparameters.momentum,
    # )
    # setattr(model.hyperparameters, 'optimizer', model.optimizer.__class__.__name__)

    # # Træn modellen 
    # model_name = train(
    #     train_loader,
    #     val_loader,
    #     model,
    # )

    # # Model er trænet, så vi gemmer den i saved_models
    # scripted_model = torch.jit.script(model)
    # scripted_model.save(f'2.NN/QuickDraw/saved_models/{model_name}.pth')