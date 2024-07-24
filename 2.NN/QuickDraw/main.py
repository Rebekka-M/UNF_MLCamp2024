from get_data import get_dataset
from data._data_static import TEGNINGER
from options import Hyperparameters, name_generator
from model import Net
from train import train


# Sæt valgmuligheder
hyperparameters = Hyperparameters(
    lr = 0.005,
)

# Hent data fra get_data.py
train_loader, val_loader = get_dataset(
    names=TEGNINGER,
    n_samples=30000,
    batch_size=hyperparameters.batch_size,
    verbose = True,
)

# Hent model architecturene fra modemll_architecture.py
model = Net(
    name = name_generator(),
    hyperparameters=hyperparameters
)

# tilføj optimizer til model
model.optimizer = model.hyperparameters.optimizer(
    model.parameters(),
    lr=model.hyperparameters.lr,
    momentum=model.hyperparameters.momentum,
)
setattr(model.hyperparameters, 'optimizer', model.optimizer.__class__.__name__)

# Træn modellen 
model_name = train(
    train_loader,
    val_loader,
    model,
)

# Model er trænet, så vi gemmer den i saved_models
model.save()