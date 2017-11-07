
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



def train(X, y):
    import linear

    w = linear.linear_regression(X, y)
    return w


def E(X, y, w):
    h = np.matmul(X, w)
    positives = h >= 0
    h[positives] = 1.0
    h[~positives] = -1.0
    return sum(h != y)[0] / len(y)



num_train = 25

X, y = prepare_data(read_matrix("in.dta"))
X_train = X[:num_train]
y_train = y[:num_train]
X_val = X[num_train:]
y_val = y[num_train:]
X_test, y_test = prepare_data(read_matrix("out.dta"))


results = []
for k in range(3, 8):
    w = train(X_train[:, :k+1], y_train[:, :k+1])
    E_val = E(X_val[:, :k+1], y_val[:, :k+1], w)
    E_out = E(X_test[:, :k+1], y_test[:, :k+1], w)
    results.append((k, E_val, E_out))


X_train, X_val = X_val, X_train
y_train, y_val = y_val, y_train
results2 = []
for k in range(3, 8):
    w = train(X_train[:, :k+1], y_train[:, :k+1])
    E_val = E(X_val[:, :k+1], y_val[:, :k+1], w)
    E_out = E(X_test[:, :k+1], y_test[:, :k+1], w)
    results2.append((k, E_val, E_out))
