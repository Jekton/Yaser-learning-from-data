
import matplotlib.pyplot as plt
import numpy as np


def compute_y_of_line(w, x):
    # w0 + w1*x + w2*y = 0
    if w[2, 0] == 0:
        return 0
    return -(w[0, 0] + w[1, 0] * x) / w[2, 0]


def plot_line(w, style = "b-"):
    xs = [-1, 2]
    ys = [compute_y_of_line(w, x) for x in xs]
    plt.plot(xs, ys, style)


def linear_regression(X, y):
    return linear_reg_weight_decay(X, y, 0)



def linear_reg_weight_decay(X, y, lambda_):
    n = X.shape[1]
    pseudo_inv = np.linalg.pinv(np.matmul(X.transpose(), X) +
                                lambda_ * np.identity(n))
    pseudo_inv = np.matmul(pseudo_inv, X.transpose())
    return np.matmul(pseudo_inv, y).reshape((n, 1))
