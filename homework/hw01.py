
import matplotlib.pyplot as plt
import numpy as np


def generate_random_line():
    line = np.random.random([2, 2])
    x1 = line[0, 0]
    y1 = line[0, 1]
    x2 = line[1, 0]
    y2 = line[1, 1]
    k = (y1 - y2) / (x1 - x2)
    b = y1 - k * x1
    # y = kx + b => -b -kx + y = 0
    f = np.empty(3)
    f[0] = -b
    f[1] = -k
    f[2] = 1
    return f.reshape((3, 1))


def compute_y_of_line(w, x):
    # w0 + w1*x + w2*y = 0
    if w[2, 0] == 0:
        return 0
    return -(w[0, 0] + w[1, 0] * x) / w[2, 0]


def plot_line(w, style = "b-"):
    xs = [-1, 2]
    ys = [compute_y_of_line(w, x) for x in xs]
    plt.plot(xs, ys, style)


def riemann_sum(w, delta):
    s = 0
    for x in np.arange(0, 1, delta):
        y = compute_y_of_line(w, x)
        if y < 0:
            y = 0
        elif y > 1:
            y = 1
        s += y * delta
    return s


def perceptron_iterate(X, y):
    N = len(y)
    w = np.zeros((3, 1))
    while True:
        done = True
        for i in range(N):
            x = X[i]
            h = 1
            if np.matmul(x, w) < 0:
                h = -1
            if h != y[i]:
                w += (y[i] * x).reshape((3, 1))
                done = False
        if done:
            break
    return w


def perceptron_random(X, y):
    bool_y = y >= 0
    N = len(y)
    w = np.zeros((3, 1))
    niter = 0
    while True:
        h = np.matmul(X, w) >= 0
        misclassify = (h != bool_y).flatten()

        if not misclassify.any():
            break

        candidates = X[misclassify]
        i = np.random.randint(candidates.shape[0])
        xi = candidates[i]
        yi = y[misclassify][i]

        h = 1
        if np.matmul(xi, w) < 0:
            h = -1
        if h != yi:
            w += (yi * xi).reshape((3, 1))
        niter += 1
    return w, niter



def simulate(N, print_res=True):
    X = np.random.random([N, 2])
    X = np.concatenate((np.ones([N, 1]), X), axis=1)

    f = generate_random_line()
    y = np.matmul(X, f)
    greaters = (y >= 0).reshape(N)
    lesses = X[~greaters]
    greaters = X[greaters]

    for i in range(N):
        if y[i] < 0:
            y[i] = -1
        else:
            y[i] = 1
    w, niter = perceptron_random(X, y)

    if print_res:
        plt.plot(greaters[:, 1], greaters[:, 2], 'rx')
        plt.plot(lesses[:, 1], lesses[:, 2], 'go')
        plot_line(f)
        plot_line(w, "r-")
        plt.xlim((-0.05, 1.05))
        plt.ylim((-0.05, 1.05))
        plt.show()
    return f, w, niter


N = 100
nexperiment = 100
delta = 0.00001
all_iters = 0
all_error = 0
for i in range(nexperiment):
    f, g, niter = simulate(N, False)
    all_iters += niter

    less_of_f = riemann_sum(f, delta)
    less_of_g = riemann_sum(g, delta)
    error = np.abs(less_of_f - less_of_g)
    all_error += error
    print(i)
print("iter =", all_iters / nexperiment)
print("error =", all_error / nexperiment)
