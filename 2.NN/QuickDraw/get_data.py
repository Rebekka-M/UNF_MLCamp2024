import os
import requests
import typing as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

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
        N_doodles = doodles.shape[0]
        X.append(doodles.reshape(N_doodles, 28, 28))
        y.append(np.full(N_doodles, i))

    # split i trænings-, og validerings-sæt med stratificering
    N = len(y)
    X = np.concatenate(X)
    y = np.concatenate(y)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size = val_size,
        random_state = seed,
        stratify = y
    )

    # konverter til torch tensors
    X_train, y_train = torch.Tensor(X_train).float(), torch.Tensor(y_train)
    X_val, y_val = torch.Tensor(X_val).float(), torch.Tensor(y_val)

    # definer datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # opret DataLoader objekter
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        sampler = RandomSampler(train_dataset, num_samples = n_samples),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        sampler = RandomSampler(val_dataset, num_samples = n_samples),
    )
    
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