import os
import random
import string
import torch
import torch.nn as nn
import mlflow
from datetime import datetime

from model_architecture import Net
from get_data import get_dataset, get_doodle_names

# Sæt mlflow tracking URI og experiment
mlflow.set_tracking_uri("")

def train(
    train_loader: torch.utils.data.DataLoader,
    model: Net):
    """
    Træner modellen
    Args:
    train_loader (torch.utils.data.DataLoader): DataLoader for træningsdata
    model (torch.nn.Module): Netværksarkitektur
    """
    # Hvis epochs ikke er specificeret, sæt til 2
    if model.hyperparameters.get('epochs') is None:
        model.hyperparameters['epochs'] = 2
    
    # Hvis lr ikke er specificeret, sæt til 0.001
    if model.hyperparameters.get('lr') is None:
        model.hyperparameters['lr'] = 0.001
    
    # Hvis momentum ikke er specificeret, sæt til 0.9
    if model.hyperparameters.get('momentum') is None:
        model.hyperparameters['momentum'] = 0.9

    # tilføj loss og optimizer
    criterion = model.criterion()
    optimizer = model.optimizer(
        model.parameters(),
        lr = model.hyperparameters['lr'], momentum=model.hyperparameters['momentum'])

    model.hyperparameters['loss'] = criterion.__class__.__name__
    model.hyperparameters['optimizer'] = optimizer.__class__.__name__

    # Check om CUDA er tilgængelig
    model_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '-' + ''.join(random.choices(string.ascii_lowercase,k=4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Start mlflow run
    with mlflow.start_run(experiment_id="0", run_name = model_name):
        # Log hyperparametre
        mlflow.log_params(model.hyperparameters)
        # Sæt model til træning
        model.train()
        
        # Trænings loop over epochs
        for epoch in range(model.hyperparameters['epochs']):
            losses = []
            accuracies = []
            val_accuracies = []
            for batch, (X, y) in enumerate(train_loader):
                X, y = X.float().to(device), y.long().to(device)

                # Genstart gradienter
                optimizer.zero_grad()

                # Forward pass
                y_hat_prob = model(X)
                y_hat = torch.argmax(y_hat_prob, dim=1).long()
                
                # Beregn loss, accuracy, og validation accuracy
                loss = criterion(y_hat_prob, y)
                losses.append(loss.item())
                accuracy = torch.sum(y_hat == y) / len(y)
                accuracies.append(accuracy)

                val_X, val_y = next(iter(val_loader))
                val_y_hat = model(val_X)
                val_accuracy = torch.sum(torch.argmax(val_y_hat, dim=1) == val_y) / len(val_y)
                val_accuracies.append(val_accuracy)

                # Backward pass og opdatering af vægte
                loss.backward()
                optimizer.step()

                # Print status
                if batch  == 0:
                    print(
                        f"loss: {loss:3f} accuracy: {accuracy:3f} [{epoch} / {model.hyperparameters['epochs']}]"
                    )

            # Log loss og accuracy
            mlflow.log_metric("loss", sum(losses) / len(losses), step=epoch)
            mlflow.log_metric("accuracy", sum(accuracies) / len(accuracies), step=epoch)
            mlflow.log_metric("val_accuracy", sum(val_accuracies) / len(val_accuracies), step=epoch)

        # Model er trænet, gem vægtene
        if not os.path.exists("2.NN/QuickDraw/saved_models"):
            os.makedirs("2.NN/QuickDraw/saved_models")
        
        torch.save(model.state_dict(), f"2.NN/QuickDraw/saved_models/{model_name}.pt")

if __name__ == "__main__":
    # Hent data fra get_data.py
    batch_size = 32
    classes = get_doodle_names()
    C = len(classes)

    train_loader, val_loader = get_dataset(
        names = classes,
        n_samples = 1000,
        batch_size = batch_size,
        verbose = True
    )

    # Hent model architecturene fra model_architecture.py
    hyperparameters = {"lr": 0.01, "momentum": 0.3, "epochs": 10}
    model = Net(C=C, hyperparameters=hyperparameters)

    # Træn modellen 
    train(train_loader, model)