
import math
import matplotlib.pyplot as plt
import numpy as np

from bin_classification import *


def E(X, y, w):
    m = X.shape[0]
    n = X.shape[1]
    e = 0
    for i in range(m):
        x = X[i, :].reshape((n, 1))
        e += math.log(1 + math.exp(-y[i] * np.matmul(w.transpose(), x)))
    return e / m;


def gradient_batch(X, y, w):
    m = X.shape[0]
    n = X.shape[1]
    g = np.ones((n, 1))
    for i in range(m):
        x = X[i, :].reshape((n, 1))
        yi = y[i]
        g += yi * x / (1 + np.exp(yi * np.matmul(w.transpose(), x)))
    return g / -m;


def gradient_stochastic(x, y, w):
        return -y * x / (1 + np.exp(y * np.matmul(w.transpose(), x)))


def diff(w0, w1):
    d = w0 - w1
    return np.sqrt(np.matmul(d.transpose(), d))


def plot_line(xs, w, style="b-"):
    import linear
    ys = [linear.compute_y_of_line(w, x) for x in xs]
    plt.plot(xs, ys, style)



def riemann_sum(w, delta):
    from linear import compute_y_of_line
    s = 0
    for x in np.arange(-1, 1, delta):
        y = compute_y_of_line(w, x)
        if y < 0:
            y = 0
        elif y > 1:
            y = 1
        s += y * delta
    return s


def simulate(N, eta, plot=False):
    sample = BinClassSample(N)
    sample.generate_sample()
    X = sample.X
    y = sample.y
    nfeature = X.shape[1]
    w = np.zeros((nfeature, 1))
    Es = []
    nstep = 0
    while True:
        indexes = np.random.permutation(N)
        w0 = w
        for i in indexes:
            x = X[i, :].reshape((nfeature, 1))
            g = gradient_stochastic(x, y[i], w)
            w = w - eta * g
        nstep += 1
        if plot:
            Es.append(E(X, y, w))
        if diff(w0, w) < 0.01:
            break

    if plot:
        print("nstep =", nstep)
        plot_line([-1, 1], w)
        sample.plot()
        plt.show()
        plt.plot(range(len(Es)), Es)
        plt.show()
    return sample.f, w, nstep



N = 100
eta = 0.01
delta = 0.0001
all_error = 0
all_step = 0
nsimulate = 100
for i in range(nsimulate):
    f, g, nstep = simulate(N, eta, False)
    less_of_f = riemann_sum(f, delta)
    less_of_g = riemann_sum(g, delta)
    error = np.abs(less_of_f - less_of_g)
    all_error += error
    all_step += nstep
    print(i)
print("step =", all_step / nsimulate)
# XXX I don't know why it's wrong
print("error =", all_error / nsimulate)
