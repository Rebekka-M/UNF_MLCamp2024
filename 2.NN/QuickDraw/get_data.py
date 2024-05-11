import os
import csv
import requests
import typing as T
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_doodle_names() -> T.List[str]:
    """
    Henter navne på labels fra Quick, Draw! dataset
    Returns:
    List[str]: Navne på labels
    """
    if os.getcwd().split('/')[-1] == 'UNF_MLCamp2024':
        path = f'2.NN/QuickDraw/data_config.csv'
    elif os.getcwd().split('/')[-1] == '2.NN':
        path = f'QuickDraw/data_config.csv'
    else:
        path = f'data_config.csv'
        
    # Hent labels
    with open(path, newline='\n') as file:
        writer = csv.reader(file)
        labels = [row[0] for row in writer]
    
    return labels

def get_doodles(name: str, verbose: bool = False) -> np.ndarray:
    """
    Downloader billeder for et bestemt label fra Quick, Draw! dataset
    Args:
    name (str): Navnet på label
    verbose (bool): Hvis True, printes status undervejs
    Returns:
    np.ndarray: Billeder for label
    """
    if os.getcwd().split('/')[-1] == 'UNF_MLCamp2024':
        path = f'2.NN/QuickDraw/data/{name}.npy'
    elif os.getcwd().split('/')[-1] == '2.NN':
        path = f'QuickDraw/data/{name}.npy'
    else:
        path = f'data/{name}.npy'
    
    # Check om data folderen allerede eksisterer
    if not os.path.exists(''.join(path.split('/')[:-1])):
        os.makedirs(''.join(path.split('/')[:-1]))

    # Check om filen allerede eksisterer
    if not os.path.exists(path):
        # Første gang vi downloader et label, skal vi hente filen fra internettet
        if verbose:
            print(f'Downloading {name}...')
        url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{name}.npy'
        r = requests.get(url, stream = True)

        # Gem filen lokalt
        with open(path, 'wb') as f:
            f.write(r.content)
    
    # Indlæs filen
    return np.load(path)

def get_dataset(
    names: T.List[str],
    n_samples: int = 1000,
    seed: int = 42,
    val_size: float = 0.2,
    batch_size: int = 32,
    verbose: bool = False,) -> torch.utils.data.Dataset:
    """
    Indlæser et dataset bestående af doodles fra Quick, Draw! dataset
    og splitter det i trænings-, validerings- og test-sæt
    Args:
    names (List[str]): Liste af labels
    n_samples (int): Antal billeder per label
    seed (int): Seed for random number generator
    val_size (float): Andel af data, der skal bruges til validering
    batch_size (int): Batch size
    verbose (bool): Hvis True, printes status undervejs
    Returns:
    :
    Trænings-, validerings- og test-sæt
    """

    # sæt seed
    np.random.seed(seed)

    # hent doodles for hvert label
    X = []
    y = []
    for i, name in enumerate(names):
        if verbose:
            print(f'Loading {name}...')
        doodles = get_doodles(name, verbose = verbose)
        # konkatener doodles og labels
        X.append(doodles[:n_samples].reshape(n_samples, 28, 28))
        y.append(np.full(n_samples, i))
    
    # Bland doodles
    X = np.concatenate(X)
    y = np.concatenate(y)
    idx = np.random.permutation(X.shape[0])
    X, y = X[idx], y[idx]
    N = X.shape[0]

    # split i et trænings-, validerings- og test-sæt
    n_train = (0, int((1 - val_size) * N))
    n_val = (n_train[1], N)

    X_train, y_train = X[n_train[0]:n_train[1]], y[n_train[0]:n_train[1]]
    X_val, y_val = X[n_val[0]:n_val[1]], y[n_val[0]:n_val[1]]

    # konverter til torch tensors
    X_train, y_train = torch.Tensor(X_train).float(), torch.Tensor(y_train)
    X_val, y_val = torch.Tensor(X_val).float(), torch.Tensor(y_val)

    # opret DataLoader objekter
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size = batch_size, shuffle = False)
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size = batch_size, shuffle = False)
    
    return train_loader, val_loader

def plot_doodles(labels: T.List[str], n_rows: int = 5, title: str = None):
    """
    Plot doodles
    Args:
    labels (List[str]): Liste af labels
    n_rows (int): Antal rækker og kolonner
    title (str): Titel på plot
    """
    # Vi skal hente N billeder
    N = n_rows ** 2
    N_per_label = N // len(labels)

    # Hent doodles for hvert label
    X = []
    y = []

    for label, name in enumerate(labels):
        doodles = get_doodles(name, verbose = False)
        X.append(doodles[:N_per_label].reshape(N_per_label, 28, 28))
        y.append(np.full(N_per_label, label))
    
    X = np.concatenate(X)
    y = np.concatenate(y)
    
    # Tegn doodles
    fig, axs = plt.subplots(n_rows, n_rows, figsize = (n_rows, n_rows), constrained_layout = True)

    for i, (X_temp, y_temp) in enumerate(zip(X, y)):
        axs[i // n_rows, i % n_rows].imshow(X_temp, cmap='Greys')
        axs[i // n_rows, i % n_rows].set_title(f"Label: {y_temp}\n({labels[y_temp]})", fontsize = 8)
        axs[i // n_rows, i % n_rows].axis('off')
    if title is None:
        title = f"{N} doodles fra Quick, Draw! dataset"
    plt.suptitle(title)
    plt.show()