import os
import random
import string
import torch
import mlflow
from datetime import datetime
from time import perf_counter

# Sæt mlflow tracking URI og experiment
mlflow.set_tracking_uri("")
def train(
    train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, model):
    """
    Træner modellen
    Args:
    train_loader (torch.utils.data.DataLoader): DataLoader for træningsdata
    val_loader (torch.utils.data.DataLoader): DataLoader for valideringsdata
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
        model.hyperparameters['momentum'] = 0.0

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
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    with mlflow.start_run(experiment_id="0", run_name = model_name):
        # Log hyperparametre
        mlflow.log_params(model.hyperparameters)
        
        # Trænings loop over epochs
        for epoch in range(model.hyperparameters['epochs']):
            losses = []
            accuracies = []
            val_losses = []
            val_accuracies = []
            # Sæt model til træning
            model.train()
            start_time = perf_counter()
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

                #val_X, val_y = next(iter(val_loader))
                #val_y_hat = model(val_X)
                #val_accuracy = torch.sum(torch.argmax(val_y_hat, dim=1) == val_y) / len(val_y)
                #val_accuracies.append(val_accuracy)

                # Backward pass og opdatering af vægte
                loss.backward()
                optimizer.step()

                # Print status
                #if batch  == 0:
                #    print(
                #        f"[{epoch} / {model.hyperparameters['epochs']}] Training:   Loss: {loss:3f} Accuracy: {accuracy:3f}"
                #    )
            model.eval()
            with torch.no_grad():
                for batch, (X, y) in enumerate(val_loader):
                    X, y = X.float().to(device), y.long().to(device)
                    y_hat_prob = model(X)
                    val_loss = criterion(y_hat_prob, y)
                    val_losses.append(val_loss.item())
                    val_accuracy = torch.sum(torch.argmax(y_hat_prob, dim=1) == y) / len(y)
                    val_accuracies.append(val_accuracy)

                    # Print status
                    #if batch  == 0:
                    #    print(
                    #        f"[{epoch} / {model.hyperparameters['epochs']}] Validation: Loss: {val_loss:3f} Accuracy: {val_accuracy:3f}"
                    #    )
            end_time = perf_counter()
            print(f"[{epoch+1} / {model.hyperparameters['epochs']} {end_time-start_time:.2f}s] Training: Loss: {sum(losses) / len(losses):3f} Accuracy: {sum(accuracies) / len(accuracies):3f} | Validation: Loss: {sum(val_losses) / len(val_losses):3f} Accuracy: {sum(val_accuracies) / len(val_accuracies):3f}")
            # Log loss og accuracy
            mlflow.log_metric("train_loss", sum(losses) / len(losses), step=epoch)
            mlflow.log_metric("train_accuracy", sum(accuracies) / len(accuracies), step=epoch)
            mlflow.log_metric("val_loss", sum(val_losses) / len(val_losses), step=epoch)
            mlflow.log_metric("val_accuracy", sum(val_accuracies) / len(val_accuracies), step=epoch)
            mlflow.log_metric("time_per_epoch", end_time - start_time, step=epoch)

        # Model er trænet, gem vægtene
        if not os.path.exists("saved_models/opgave1/"):
            os.makedirs("saved_models/opgave1/")
        
        torch.save(model.state_dict(), f"saved_models/opgave1/{model_name}.pt")
        print(f"Gemt modelen i saved_models/opgave1/{model_name}.pt")