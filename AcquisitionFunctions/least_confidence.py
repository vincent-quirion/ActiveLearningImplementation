import numpy as np
from torch.utils.data import DataLoader, Subset

from utils.compute_probs import compute_probs

batch_size = 1000


def least_confidence(d_avail, n, train_dataset, model):
    subset = Subset(train_dataset, np.arange(len(d_avail))[d_avail])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    probs = compute_probs(loader, model)

    to_label = subset.indices[probs.max(1)[0].sort()[1][:n]]
    d_avail[to_label] = False

    return d_avail
