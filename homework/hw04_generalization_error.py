
import matplotlib.pyplot as plt
import numpy as np

"""
Break log(mH(N)), otherwise, it will overflow
"""


def vc_bound(delta, d, N):
    return np.sqrt(8 / N * (np.log(4 / delta) + d * np.log(2 * N)))


def rademacher_penalty(delta, d, N):
    ret = np.sqrt(2 / N * (np.log(2) + (d + 1) * np.log(N)))
    ret += np.sqrt(2 / N * np.log(1 / delta))
    return ret + 1 / N


def parrondo(delta, d, N):
    def do_parrondo(epsilon):
        ret = np.sqrt(1 / N * (2 * epsilon + np.log(6) + d * np.log(2 * N)))
        return ret;
    epsilon = 0
    for i in range(20):
        epsilon = do_parrondo(epsilon)
    return epsilon


def devroye(delta, d, N):
    def do_devroye(epsilon):
        return np.sqrt(1 / (2 * N) * (4 * epsilon * (1 + epsilon) + np.log(4 / delta) + 2 * d * np.log(N)))
    epsilon = 0
    for i in range(20):
        epsilon = do_devroye(epsilon)
    return epsilon



delta = 0.05
d = 50
xs = np.linspace(1, 10000, num=100)

plt.plot(xs, vc_bound(delta, d, xs), "-", label="vc_bound")
plt.plot(xs, rademacher_penalty(delta, d, xs), "-", label="rademacher_penalty")
plt.plot(xs, parrondo(delta, d, xs), "-", label="parrondo")
plt.plot(xs, devroye(delta, d, xs), "-", label="devroye")
plt.legend()

#plt.ylim((0, 0.5))

plt.show()


N = 5
print("vc_bound =", vc_bound(delta, d, N))
print("rademacher_penalty =", rademacher_penalty(delta, d, N))
print("parrondo =", parrondo(delta, d, N))
print("devroye =", devroye(delta, d, N))
