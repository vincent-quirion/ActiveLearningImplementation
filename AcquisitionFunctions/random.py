import numpy as np


def random(d_avail, n, **kwargs):
    to_label = np.random.choice(np.flatnonzero(d_avail), size=int(n), replace=False)

    d_avail[to_label] = False

    return d_avail
