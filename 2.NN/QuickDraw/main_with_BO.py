from get_data import get_dataset
from data._data_static import TEGNINGER
from options import Hyperparameters, name_generator
from model import Net
from train import train

import optuna
from torch.optim import Adam, SGD, AdamW

def objective(trial: optuna.Trial) -> float:
    # Sæt valgmuligheder
    optimizer = trial.suggest_categorical('optimizer', ["Adam", "SGD", "AdamW"])

    if optimizer in ["Adam", "AdamW"]:
        hyperparameters = Hyperparameters(
            lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True),
            optimizer = Adam if optimizer == "Adam" else AdamW,
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            betas=(
                trial.suggest_float("beta1", 0.1, 0.9),
                trial.suggest_float("beta2", 0.9, 0.999),
            ),
            eps=trial.suggest_float("eps", 1e-8, 1e-6, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            epochs=25,
        )
    else:
        hyperparameters = Hyperparameters(
            lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True),
            optimizer = SGD,
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            momentum = trial.suggest_float("momentum", 0.0, 0.9),
            dampening = trial.suggest_float("dampening", 0.0, 0.9),
            epochs=25,
        )

    # Hent data fra get_data.py
    train_loader, val_loader = get_dataset(
        names=TEGNINGER,
        n_samples=30000,
        batch_size=hyperparameters.batch_size,
        verbose = True,
    )

    # Hent model architecturene fra model.py
    model = Net(
        name = name_generator(),
        hyperparameters=hyperparameters,
        n_capsules=trial.suggest_categorical("n_capsules", [16, 32, 64]),
    )

    # tilføj optimizer til model
    if optimizer in ["Adam", "AdamW"]:
        model.optimizer = model.hyperparameters.optimizer(
            model.parameters(),
            lr=model.hyperparameters.lr,
            betas=model.hyperparameters.betas,
            eps=model.hyperparameters.eps,
            weight_decay=model.hyperparameters.weight_decay,
        )
    else:
        model.optimizer = model.hyperparameters.optimizer(
            model.parameters(),
            lr=model.hyperparameters.lr,
            momentum=model.hyperparameters.momentum,
            dampening=model.hyperparameters.dampening,
            weight_decay=model.hyperparameters.weight_decay,
        )
    setattr(model.hyperparameters, 'optimizer', model.optimizer.__class__.__name__)

    # Træn modellen 
    validation_accuracy = train(
        train_loader,
        val_loader,
        model,
    )

    # Model er trænet, så vi gemmer den i saved_models
    model.save()

    return validation_accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)