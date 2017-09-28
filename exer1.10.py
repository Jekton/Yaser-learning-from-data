
import numpy as np
import matplotlib.pyplot as plt


def simulate(ncoin, ntoss):
    outcome = np.empty((ncoin, ntoss))
    for i in range(ncoin):
        for j in range(ntoss):
            outcome[i, j] = np.random.choice((0, 1));
    heads = np.sum(outcome, axis=1)
    heads /= ntoss
    return heads[0], np.random.choice(heads), np.min(heads)
