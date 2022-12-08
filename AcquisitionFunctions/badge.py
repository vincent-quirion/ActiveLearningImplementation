import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from train import loss_fn
from utils.kmeans_plusplus import kmeans_plusplus


def badge(d_avail, n, train_dataset, model):
    subset = Subset(train_dataset, np.arange(len(d_avail))[d_avail])

    loader = DataLoader(subset, batch_size=1, shuffle=False)

    gradient_embeddings = torch.zeros((len(subset), *list(model.parameters())[-2].shape))

    model.eval()

    # Disable autograd for every paramter except the weights of the last layer for a ≈6x speedup
    model_parameters = list(model.parameters())
    for i, param in enumerate(model_parameters):
        if i != len(model_parameters) - 2:
            param.requires_grad = False

    print("Computing last layer weights' gradients")
    for i, (x, y) in enumerate(loader):
        model.zero_grad()

        out = model(x)

        # In practice, we don't have access to the true label so we use the model's prediction instead
        fake_label = F.one_hot(torch.max(out, dim=1).indices, num_classes=out.shape[1]).float()

        loss = loss_fn(out, fake_label)
        loss.backward()

        gradient_embeddings[i] = list(model.parameters())[-2].grad

    # Turn autograd back on
    for i, param in enumerate(model_parameters):
        if i != len(model_parameters) - 2:
            param.requires_grad = True

    # Find centroids with K-Means++
    centroids = kmeans_plusplus(gradient_embeddings.flatten(start_dim=1), n)

    d_avail[subset.indices[centroids]] = False

    return d_avail
