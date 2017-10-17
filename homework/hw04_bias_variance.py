
import numpy as np
import matplotlib.pyplot as plt

import linear


def fmin(X, y, theta, func, alpha, steps, plot=False):
    xs = range(steps)
    Js = []
    for x in xs:
        J, gradient = func(X, y, theta)
        theta += -alpha * gradient
        Js.append(J)
    if plot:
        plt.plot(xs, Js)
        plt.show()
    return theta



def cost_linear(X, y, theta):
    m = X.shape[0]
    h = np.matmul(X, theta)
    delta = h - y
    J = np.matmul(delta.transpose(), delta)[0] / (2 * m)
    gradient = np.matmul(X.transpose(), delta) / m;
    return J, gradient


def cost_ax(X, y, theta):
    assert theta[0] == 0
    J, gradient = cost_linear(X, y, theta)
    gradient[0] = 0
    return J, gradient




def f(x):
    return np.sin(np.pi * x)


def gen_dataset():
    m = 2  # data set size
    low, high = -1, 1
    X = np.random.uniform(low, high, (m, 1))
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    y = f(X[:, 1]).reshape((m, 1))
    return X, y


def train_ax(plot=False):
    n = 1  # feature size
    X, y = gen_dataset()
    theta = np.zeros((n + 1, 1))
    theta = fmin(X, y, theta, cost_ax, 0.1, 1000, plot)

    x1 = X[0, 1]
    x2 = X[1, 1]
    y1 = y[0]
    y2 = y[1]
    theory = (x1 * y1 + x2 * y2) / (x1**2 + x2**2)

    if plot:
        plt.plot(0, 0, "ro")
        plt.plot(X[:, 1], y, "o")
        w = np.array((0, theta[1], -1))
        w = w.reshape((3, 1))
        linear.plot_line(w, "g-")
        w[1] = theory
        linear.plot_line(w, "r-")
        plt.xlim((-1, 1))
        plt.ylim((-1, 1))
        plt.show()
    return theta, theory


def train_b():
    X, y = gen_dataset()
    return (y[0] + y[1]) / 2


def train_ax_plus_b():
    n = 1  # feature size
    X, y = gen_dataset()
    theta = np.zeros((n + 1, 1))
    theta = fmin(X, y, theta, cost_linear, 0.3, 400)
    return theta


def train_ax_square():
    n = 1  # feature size
    X, y = gen_dataset()
    X[:, 1] = X[:, 1] * X[:, 1]
    theta = np.zeros((n + 1, 1))
    theta = fmin(X, y, theta, cost_ax, 0.3, 400)
    return theta


def train_ax_square_plus_b():
    n = 1  # feature size
    X, y = gen_dataset()
    X[:, 1] = X[:, 1] * X[:, 1]
    theta = np.zeros((n + 1, 1))
    theta = fmin(X, y, theta, cost_linear, 0.3, 400)
    return theta


def bias_variance(func, params, epsilon):
    npoint = 2 / epsilon
    xs = np.linspace(-1, 1, npoint)
    y = f(xs)

    param_bar = sum(params) / len(params)
    h = func(param_bar, xs)
    err = y - h
    bias = np.dot(err, err) / npoint
    print("bias =", bias)

    variance = 0
    i = 0
    for param in params:
        hh = func(param, xs)
        err = h - hh
        variance += np.dot(err, err) / npoint
        print(i)
        i += 1
    variance /= len(params)
    print("variance =" , variance)
    return bias, variance

def hypo_ax(N, epsilon):
    #train_ax(True)
    thetas = []
    for i in range(N):
        theta, theory = train_ax()
        thetas.append(theta)
        print(i)
    a = sum(thetas) / len(thetas)
    print("a =", a[1])

    def g(param, x):
        a = param[1]
        return a * x
    return bias_variance(g, thetas, epsilon)


def hypo_b(N, epsilon):
    bs = []
    for i in range(N):
        b = train_b()
        bs.append(b)
        print(i)
    def g(param, x):
        return param
    return bias_variance(g, bs, epsilon)


def hypo_ax_plus_b(N, epsilon):
    thetas = []
    for i in range(N):
        theta = train_ax_plus_b()
        thetas.append(theta)
        print(i)

    def g(param, x):
        a = param[1]
        b = param[0]
        return a * x + b
    return bias_variance(g, thetas, epsilon)


def hypo_ax_square(N, epsilon):
    thetas = []
    for i in range(N):
        theta = train_ax_square()
        thetas.append(theta)
        print(i)

    def g(param, x):
        a = param[1]
        return a * x * x
    return bias_variance(g, thetas, epsilon)



def hypo_ax_square_plus_b(N, epsilon):
    thetas = []
    for i in range(N):
        theta = train_ax_square()
        thetas.append(theta)
        print(i)

    def g(param, x):
        a = param[1]
        b = param[0]
        return a * x * x + b
    return bias_variance(g, thetas, epsilon)
