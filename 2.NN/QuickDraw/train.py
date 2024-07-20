"""Modul til træning af neurale netværk"""

import torch
import mlflow
from torch import nn
from dataclasses import asdict

# Sæt mlflow tracking URI og experiment
mlflow.set_tracking_uri("")

def train(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    ) -> str:
    """
    Træner modellen
    Args:
    train_loader (torch.utils.data.DataLoader): DataLoader for træningsdata
    val_loader (torch.utils.data.DataLoader): DataLoader for valideringsdata
    model (torch.nn.Module): Netværksarkitektur
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Start mlflow run
    with mlflow.start_run(experiment_id="0", run_name = model.name):
        # Log hyperparametre
        mlflow.log_params(asdict(model.hyperparameters))
        # Sæt model til træning
        model.train()
        
        # Trænings loop over epochs
        for epoch in range(model.hyperparameters.epochs):
            losses = []
            accuracies = []
            val_accuracies = []
            for batch, (X, y) in enumerate(train_loader):
                X, y = X.float().to(device), y.long().to(device)

                # Genstart gradienter
                model.optimizer.zero_grad()

                # Forward pass
                y_hat_prob = model(X)
                y_hat = torch.argmax(y_hat_prob, dim=1).long()
                
                # Beregn loss, accuracy, og validation accuracy
                loss = model.criterion(y_hat_prob, y)
                losses.append(loss.item())
                accuracy = torch.sum(y_hat == y) / len(y)
                accuracies.append(accuracy)

                val_X, val_y = next(iter(val_loader))
                val_y_hat = model(val_X)
                val_accuracy = torch.sum(torch.argmax(val_y_hat, dim=1) == val_y) / len(val_y)
                val_accuracies.append(val_accuracy)

                # Backward pass og opdatering af vægte
                loss.backward()
                model.optimizer.step()

                # Print status
                if batch == 0:
                    print(
                        f"loss: {loss:3f} accuracy: {accuracy:3f} [{epoch} / {model.hyperparameters.epochs}]"
                    )

            # Log loss og accuracy
            mlflow.log_metric("loss", sum(losses) / len(losses), step=epoch)
            mlflow.log_metric("accuracy", sum(accuracies) / len(accuracies), step=epoch)
            mlflow.log_metric("val_accuracy", sum(val_accuracies) / len(val_accuracies), step=epoch)