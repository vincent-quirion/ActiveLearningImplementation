import torch
import torch.nn.functional as F


def compute_probs(loader, model):
    probs = torch.zeros([len(loader.dataset), 10])

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            out = model(x)
            out = F.softmax(out, dim=1)
            probs[loader.batch_size * i : loader.batch_size * i + len(x)] = out

    return probs
