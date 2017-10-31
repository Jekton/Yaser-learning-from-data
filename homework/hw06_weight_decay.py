
import math
import matplotlib.pyplot as plt
import numpy as np


def read_matrix(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
        m = len(lines)
        if m == 0:
            return None
        n = len(lines[0].split())
        mat = np.empty((m, n))
        for i, line in enumerate(lines):
            for j, number in enumerate(line.split()):
                mat[i][j] = float(number)
        return mat
    return None



def prepare_data(mat):
    m = mat.shape[0]
    n = 8
    X = np.ones((m, n))
    y = np.empty((m, 1))
    y = mat[:, 2]
    y = y.reshape((m, 1))
    x1 = mat[:, 0]
    x2 = mat[:, 1]
    X[:, 1] = x1
    X[:, 2] = x2
    X[:, 3] = x1 ** 2
    X[:, 4] = x2 ** 2
    X[:, 5] = x1 * x2
    X[:, 6] = np.abs(x1 - x2)
    X[:, 7] = np.abs(x1 + x2)
    return X, y



def train(X, y, lambda_):
    import linear
    w = linear.linear_regression(X, y)
    w_reg = linear.linear_reg_weight_decay(X, y, lambda_)
    return w, w_reg


def E(X, y, w):
    h = np.matmul(X, w)
    positives = h >= 0
    h[positives] = 1.0
    h[~positives] = -1.0
    return sum(h != y)[0] / len(y)




X, y = prepare_data(read_matrix("in.dta"))
X_test, y_test = prepare_data(read_matrix("out.dta"))
w, w_reg = train(X, y, 1e-3)
print("E_in = %f, E_out = %f" % (E(X, y, w), E(X_test, y_test, w)))
print("E_in = %f, E_out = %f" % (E(X, y, w_reg), E(X_test, y_test, w_reg)))



ks = [2, 1, 0, -1, -2]
E_smallest = 100
winner = 0
for k in ks:
    lambda_ = 10 ** k
    w, w_reg = train(X, y, lambda_)
    E_out = E(X_test, y_test, w_reg)
    if E_out < E_smallest:
        winner = k
        E_smallest = E_out
    print("E_out = %f, k = %f" % (E_out, k))
print("winner =", winner)
