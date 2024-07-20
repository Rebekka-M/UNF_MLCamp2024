"""Dataclasses til hyperparameters, training, og data valgmuligheder"""
import random
import string
from datetime import datetime
import typing as T
from dataclasses import dataclass
from torch import optim    

@dataclass
class Hyperparameters:
    # data parameters
    batch_size: T.Union[int, T.List[int]] = 32

    # training parametre
    epochs: T.Union[int, T.List[int]] = 10

    # optimizer parametre
    optimizer: optim.Optimizer = optim.SGDb
    lr: T.Union[float, T.List[float]] = 0.001
    betas: T.Tuple[float] = (0.9, 0.999)
    dampening: float = 0
    eps: float = 1e-08,
    momentum: T.Union[float, T.List[float]] = 0.5
    nesterov: bool = False
    weight_decay: float = 0

def name_generator():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '-' + ''.join(random.choices(string.ascii_lowercase,k=4))
