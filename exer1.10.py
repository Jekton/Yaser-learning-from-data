
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



ncoin = 1000
ntoss = 10
mu = 0.5
nexperiment = 10000

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


def P(outcomes, mu, epsilon):
    less = np.sum(outcomes < mu - epsilon)
    greater = np.sum(outcomes > mu + epsilon)
    return (less + greater) / len(outcomes)

def hoeffding(epsilon, N):
    return 2 * np.exp(-2 * epsilon * epsilon * N)

epsilons = np.linspace(0, 1, 100)
P_first = [P(experiments[:, 0], mu, epsilon) for epsilon in epsilons]
P_random = [P(experiments[:, 1], mu, epsilon) for epsilon in epsilons]
P_min = [P(experiments[:, 2], mu, epsilon) for epsilon in epsilons]
P_hoeffding = [hoeffding(epsilon, ntoss) for epsilon in epsilons]

plt.plot(epsilons, P_first, label="first")
plt.plot(epsilons, P_random, label="random")
plt.plot(epsilons, P_min, label="min")
plt.plot(epsilons, P_hoeffding, label="Hoeffding")
plt.legend()
plt.show()

