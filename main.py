import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST

from AcquisitionFunctions.least_confident import least_confidence
from AcquisitionFunctions.random import random
from eval_model import eval_model
from model import resnet18MNIST
from plot import Experience
from train import train
from utils import transform

initially_labeled = 2000
label_batch_size = 1000
max_steps = 8
epochs = 4

experience = Experience()

train_dataset = MNIST("data/mnist", train=True, download=True, transform=transform)
test_dataset = MNIST("data/mnist", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64)

d_avail = np.ones(len(train_dataset), dtype=bool)

model = resnet18MNIST()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
)
# Save initial model parameters
torch.save(model.state_dict(), "models/initial_model.pt")

# First iteration is always from the random acquisition function
init_avail = random(d_avail, initially_labeled)

for name, acquisition_fn in [("Random", random), ("Least Confident", least_confidence)]:
    # Reset model
    model.load_state_dict(torch.load("models/initial_model.pt"))
    # Start with initially labeled examples
    d_avail = init_avail.copy()
    # Train with initially labeled examples
    train(
        model,
        DataLoader(Subset(train_dataset, np.arange(len(train_dataset))[~d_avail]), batch_size=64, shuffle=True),
        optimizer,
        epochs=epochs,
        acquisition_function=name,
        step=0,
        max_steps=max_steps,
    )
    # Test accuracy with initially labeled examples
    experience.add_test(
        name, initially_labeled, eval_model(model, test_loader, acquisition_function=name, step=0, max_steps=max_steps)
    )

    for step in range(max_steps):
        d_avail = acquisition_fn(d_avail, label_batch_size, train_dataset=train_dataset, model=model)
        train(
            model,
            DataLoader(
                Subset(train_dataset, np.arange(len(train_dataset))[~d_avail]),
                batch_size=64,
            ),
            optimizer,
            epochs=epochs,
            acquisition_function=name,
            step=step + 1,
            max_steps=max_steps,
        )
        experience.add_test(
            name,
            initially_labeled + label_batch_size * (step + 1),
            eval_model(model, test_loader, acquisition_function=name, step=step + 1, max_steps=max_steps),
        )


experience.save_test_plot(f"plots/random_vs_lc_{initially_labeled}_initial_{label_batch_size}_label_batch_size")
