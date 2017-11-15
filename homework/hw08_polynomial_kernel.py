
import numpy as np
from sklearn.svm import SVC


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


def train_one_vs_all(X, y, train_who, C, Q):
    y = np.copy(y)
    target_index = y == train_who
    y[target_index] = 1
    y[~target_index] = -1
    classifer = SVC(C=C, kernel='poly', gamma=1, coef0=1, degree=Q)
    classifer.fit(X, y)
    return classifer.score(X, y), len(classifer.support_vectors_)


def train_one_vs_one(X, y, train0, train1, C, Q):
    target0 = y == train0
    target1 = y == train1
    X0 = X[target0]
    X1 = X[target1]
    y0 = y[target0]
    y1 = y[target1]
    X = np.concatenate((X0, X1), axis=0)
    y = np.concatenate((y0, y1), axis=0)
    classifer = SVC(C=C, kernel='poly', gamma=1, coef0=1, degree=Q)
    classifer.fit(X, y)
    return classifer.score(X, y), len(classifer.support_vectors_)


X, y = prepare_data(read_matrix('./features.train'))

# Problem 2~4
# C = 0.01
# Q = 2
# scores = []
# for class_ in range(10):
#     score, support_vectors = train_one_vs_all(X, y, class_, C, Q)
#     print(class_, score, support_vectors)
#     scores.append(score)
# print('min =', min(scores))
# print('max =', max(scores))

C = [0.0001, 0.001, 0.01, 0.1, 1]
Q = [2, 5]
for c in C:
    for q in Q:
        score, support_vectors = train_one_vs_one(X, y, 1, 5, c, q)
        print('C = %.4f, Q = %d: %.20f, %d' % (c, q, score, support_vectors))
