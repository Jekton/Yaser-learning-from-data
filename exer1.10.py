
import numpy as np
import matplotlib.pyplot as plt


def simulate(ncoin, ntoss):
    outcome = np.empty((ncoin, ntoss))
    for i in range(ncoin):
        for j in range(ntoss):
            outcome[i, j] = np.random.choice((0, 1));
    heads = np.sum(outcome, axis=1)
    heads /= 10
    return heads[0], np.random.choice(heads), np.min(heads)

def hoeffding(mu, nu, N):
    epsilon = np.abs(mu - nu)
    return 2 * np.exp(-2 * epsilon * epsilon * N)


ncoin = 1000
ntoss = 10
mu = 0.5
nexperiment = 100

experiments = np.empty((nexperiment, 3))
for i in range(nexperiment):
    first, rand, minimun = simulate(ncoin, ntoss)
    experiments[i, 0], experiments[i, 1], experiments[i, 2] = first, rand, minimun
    print(i)

bins = np.linspace(0, 1, 22)
plt.hist(experiments[:, 0], bins, alpha=0.5, label="first one")
plt.hist(experiments[:, 1], bins, alpha=0.5, label="random")
plt.hist(experiments[:, 2], bins, alpha=0.5, label="minimun")
plt.legend()
plt.show()


