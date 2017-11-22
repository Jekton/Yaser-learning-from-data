
import numpy as np

import linear


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
    num_data = mat.shape[0]
    num_feature = mat.shape[1]
    X = np.ones((num_data, num_feature))
    y = np.empty(num_data)
    y = mat[:, 0]
    X[:, 1] = mat[:, 1]
    X[:, 2] = mat[:, 2]
    return X, y


def prepare_z_space_data(X):
    num_data = X.shape[0]
    Z = np.ones((num_data, 6))
    x1 = X[:, 1]
    x2 = X[:, 2]
    Z[:, 1] = x1
    Z[:, 2] = x2
    Z[:, 3] = x1 * x2
    Z[:, 4] = x1**2
    Z[:, 5] = x2**2
    return Z


def prepare_one_vs_all(y, target):
    y = np.copy(y)
    target_index = y == target
    y[target_index] = 1
    y[~target_index] = -1
    return y


def one_vs_all(X, y, X_test, y_test, target, lambda_):
    y = prepare_one_vs_all(y, target)
    y_test = prepare_one_vs_all(y_test, target)
    w = linear.linear_reg_weight_decay(X, y.reshape((-1, 1)), lambda_)
    pred = np.matmul(X_test, w).flatten()
    positives = pred > 0
    pred[positives] = 1
    pred[~positives] = -1
    return np.mean(pred == y_test)


def prepare_one_vs_one(X, y, target0, target1):
    y = np.copy(y)
    target0 = y == target0
    target1 = y == target1
    X0 = X[target0]
    X1 = X[target1]
    y0 = y[target0]
    y1 = y[target1]
    y0.fill(1)
    y1.fill(-1)
    X = np.concatenate((X0, X1), axis=0)
    y = np.concatenate((y0, y1), axis=0)
    return X, y


def one_vs_one(X, y, X_test, y_test, target0, target1, lambda_):
    X, y = prepare_one_vs_one(X, y, target0, target1)
    X_test, y_test = prepare_one_vs_one(X_test, y_test, target0, target1)
    w = linear.linear_reg_weight_decay(X, y.reshape((-1, 1)), lambda_)
    pred = np.matmul(X_test, w).flatten()
    positives = pred > 0
    pred[positives] = 1
    pred[~positives] = -1
    return np.mean(pred == y_test)


X_train, y_train = prepare_data(read_matrix('./features.train'))
X_test, y_test = prepare_data(read_matrix('./features.test'))
Z_train = prepare_z_space_data(X_train)
Z_test = prepare_z_space_data(X_test)

lambda_ = 1

# print('Problem 7')
# for target in range(5, 10):
#     print(target, one_vs_all(X_train, y_train, X_train, y_train,
#                              target, lambda_))

# print('Problem 8')
# for target in range(5):
#     print(target, one_vs_all(Z_train, y_train, Z_test, y_test,
#                              target, lambda_))

# print('Problem 9')
# for target in range(10):
#     print(target, 'E_in without transform',
#           one_vs_all(X_train, y_train, X_train, y_train,
#                      target, lambda_))
#     print(target, 'E_in with transform',
#           one_vs_all(Z_train, y_train, Z_train, y_train,
#                      target, lambda_))
#     print(target, 'E_out without transform',
#           one_vs_all(X_train, y_train, X_test, y_test,
#                      target, lambda_))
#     print(target, 'E_out with transform',
#           one_vs_all(Z_train, y_train, Z_test, y_test,
#                      target, lambda_))
#     print('------------')

print('Problem 10')
target0 = 1
target1 = 5
E_in_1 = None
E_out_1 = None
for lambda_ in (0.01, 1):
    E_in_z = one_vs_one(Z_train, y_train, Z_train, y_train,
                        target0, target1, lambda_)
    E_out_z = one_vs_one(Z_train, y_train, Z_test, y_test,
                         target0, target1, lambda_)
    print(lambda_, 'E_in', '%.40f' % E_in_z)
    print(lambda_, 'E_out', '%.40f' % E_out_z)
    print('------------')
    if E_in_1 is None:
        E_in_1 = E_in_z
        E_out_1 = E_out_z
    else:
        print('E_in is equal', E_in_1 == E_in_z)
        print('E_out is equal', E_out_1 == E_out_z)
