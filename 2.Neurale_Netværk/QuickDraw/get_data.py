import os
import io
import requests
import numpy as np
import matplotlib.pyplot as plt

def get_doodles(name: str, verbose: bool = False):
    if not os.path.exists(f'data/{name}.npy'):
        if verbose:
            print(f'Downloading {name}...')
        url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{name}.npy'
        r = requests.get(url, stream = True)
        with open(f'data/{name}.npy', 'wb') as f:
            f.write(r.content)
    
    return np.load(f'data/{name}.npy')

def get_dataset(names: list,
                n_samples: int = 1000,
                verbose: bool = False,
                seed: int = 42,
                test_size: float = 0.2,
                val_size: float = 0.2):
    np.random.seed(seed)
    X = []
    y = []
    
    for i, name in enumerate(names):
        if verbose:
            print(f'Loading {name}...')
        doodles = get_doodles(name, verbose = verbose)
        X.append(doodles[:n_samples].reshape(n_samples, 28, 28))
        y.append(np.full(n_samples, i))
    
    # shuffle
    X = np.concatenate(X)
    y = np.concatenate(y)
    idx = np.random.permutation(X.shape[0])
    X, y = X[idx], y[idx]
    N = X.shape[0]

    # split into training, validation, and test sets
    n_train = (0, int((1 - test_size - val_size) * N))
    n_val = (n_train[1], n_train[1] + int(val_size * N))
    n_test = (n_val[1], N)

    X_train, y_train = X[n_train[0]:n_train[1]], y[n_train[0]:n_train[1]]
    X_val, y_val = X[n_val[0]:n_val[1]], y[n_val[0]:n_val[1]]
    X_test, y_test = X[n_test[0]:n_test[1]], y[n_test[0]:n_test[1]]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def plot_doodles(X, y, label_dict, n_rows: int = 5, title: str = None):
    fig, axs = plt.subplots(n_rows, n_rows, figsize = (n_rows * 2, n_rows * 2))
    N = n_rows ** 2
    for i, (X_temp, y_temp) in enumerate(zip(X[:N], y[:N])):
        axs[i // n_rows, i % n_rows].imshow(X_temp, cmap='Greys')
        axs[i // n_rows, i % n_rows].set_title(f"Label: {y_temp} ({label_dict[y_temp]})")
        axs[i // n_rows, i % n_rows].axis('off')
    if title is None:
        title = f"{N} doodles fra Quick, Draw! dataset"
    plt.suptitle(title)
    plt.show()