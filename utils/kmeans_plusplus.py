import numpy as np
from scipy import stats
from sklearn.metrics import pairwise_distances
from torch import is_tensor


def kmeans_plusplus(X, K):
    if is_tensor(X):
        X = X.numpy()

    return _kmeans_plusplus(X, K)


"""
K-means++ initialization (returns the index of the centroids)
Implementation from
https://github.com/JordanAsh/badge/blob/2501e7fbd82a8e948f90df6e9f1684a40a6841e2/query_strategies/badge_sampling.py#L46
because the sklearn implementation can return duplicate centroids which is undesirable.
"""


def _kmeans_plusplus(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.0] * len(X)
    cent = 0
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]

        D2 = D2.ravel().astype(float)
        Ddist = (D2**2) / sum(D2**2)
        customDist = stats.rv_discrete(name="custm", values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll:
            ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll
