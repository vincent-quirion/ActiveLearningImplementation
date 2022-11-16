import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

batch_size = 1000


def least_confidence(d_avail, n, train_dataset, model):
    probs = torch.zeros([len(np.flatnonzero(d_avail)), 10])
    subset = Subset(train_dataset, np.arange(len(d_avail))[d_avail])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            out = model(x)
            out = F.softmax(out, dim=1)
            probs[np.arange(batch_size * i, min(batch_size * (i + 1), len(probs)))] = out

    to_label = subset.indices[probs.max(1)[0].sort()[1][:n]]
    d_avail[to_label] = False

    return d_avail
