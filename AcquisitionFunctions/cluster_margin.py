from math import ceil

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader, Subset
from torchvision.models.feature_extraction import create_feature_extractor

from utils.compute_probs import compute_probs

cluster_labels = None


def _init_clusters(train_dataset, model):
    # Extract the inputs' embeddings from the model's last/"penultimate" layer
    batch_size = 128
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    extractor = create_feature_extractor(model, ["layer4.1.relu_1"])

    # The model's "penultimate" layer has 512 outputs
    outs = np.zeros((len(train_dataset), 512))

    for i, (x, y) in enumerate(loader):
        outs[batch_size * i : batch_size * i + len(x)] = (
            extractor(x)["layer4.1.relu_1"].reshape((len(x), 512)).detach().numpy()
        )

    # Create clusters with HAC
    # Original paper doesn't use a set number of clusters
    # but instead chooses an average distance that produces an average of 10 clusters
    print("Clustering")
    clustering = AgglomerativeClustering(n_clusters=10, linkage="average").fit(outs)

    return clustering.labels_


def cluster_margin(d_avail, n, train_dataset, model):
    global cluster_labels

    # To ensure example diversity, we sample more examples than needed (to not saturate clusters as quickly during round-robin sampling)
    # See page 4 of https://arxiv.org/pdf/2107.14263.pdf
    k_m = ceil(n * 1.2)

    if cluster_labels is None:
        cluster_labels = _init_clusters(train_dataset, model)

    # Compute the margin score of each example
    batch_size = 128
    subset = Subset(train_dataset, np.arange(len(d_avail))[d_avail])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    probs = compute_probs(loader, model)

    def compute_margin(t):
        sorted_t = t.sort()[0]
        return sorted_t[-1] - sorted_t[-2]

    margin_scores_indexes = torch.Tensor([compute_margin(t) for t in probs]).sort()[1][:k_m]

    # Map selected examples to their cluster
    # (examples will be sorted in ascending order in the clusters)
    clusters = [[]] * len(np.unique(cluster_labels))
    for example_index in margin_scores_indexes:
        real_index = subset.indices[example_index]
        cluster = cluster_labels[real_index]
        clusters[cluster].append(real_index)

    clusters.sort(key=len)

    # Round-robin sampling
    added_examples = 0
    cluster_index = 0
    while added_examples < n:
        if len(clusters[cluster_index]) > 0:
            d_avail[clusters[cluster_index].pop(0)] = False
            cluster
            added_examples += 1
        cluster_index = cluster_index + 1 if cluster_index < len(clusters) - 1 else 0

    return d_avail
