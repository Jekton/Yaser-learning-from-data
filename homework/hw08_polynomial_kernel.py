
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

# XXX Problem 2~6 must consider the test set
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


# Problem 5~6
# C = [0.0001, 0.001, 0.01, 0.1, 1]
# Q = [2, 5]
# for c in C:
#     for q in Q:
#         score, support_vectors = train_one_vs_one(X, y, 1, 5, c, q)
#         print('C = %.4f, Q = %d: %.20f, %d' % (c, q, score, support_vectors))


def prepare_one_vs_one(X, y, target0, target1):
    t0 = y == target0
    t1 = y == target1
    X0 = X[t0]
    X1 = X[t1]
    y0 = y[t0]
    y1 = y[t1]
    X = np.concatenate((X0, X1), axis=0)
    y = np.concatenate((y0, y1), axis=0)
    return X, y


def train_cv(X, y, val_start, val_end, C, Q):
    X_train = np.concatenate((X[0:val_start], X[val_end:]), axis=0)
    y_train = np.concatenate((y[0:val_start], y[val_end:]))
    X_val = X[val_start:val_end]
    y_val = y[val_start:val_end]
    classifer = SVC(C=C, kernel='poly', gamma=1, coef0=1, degree=Q)
    classifer.fit(X_train, y_train)
    return classifer.score(X_val, y_val)


def shuffle(X, y):
    num_data = X.shape[0]
    shuffle_indices = np.random.choice(num_data, size=num_data, replace=True)
    return X[shuffle_indices], y[shuffle_indices]


def cross_validation(X, y, C, Q):
    num_test_set = X.shape[0]
    num_val = int(num_test_set / 10)
    total_scores = 0
    val_start = 0
    for i in range(10):
        val_end = val_start + num_val
        score = train_cv(X, y, val_start, val_end, C, Q)
        total_scores += score
        val_start = val_end
    return total_scores / 10


# Problem 7~8 (but is wrong)
# C = [0.0001, 0.001, 0.01, 0.1, 1]
# Q = 2
# d = {}
# X, y = prepare_one_vs_one(X, y, 1, 5)
# scores = np.zeros(len(C))
# for i in range(500):
#     selected = -1
#     max_score = 0
#     X, y = shuffle(X, y)
#     for j, c in enumerate(C):
#         score = cross_validation(X, y, c, Q)
#         scores[j] += score
#         if score > max_score:
#             max_score = score
#             selected = c
#     d[selected] = d.get(selected, 0) + 1
#     print(i)
# scores /= 500
# print(d)
# print(1 - scores[0])



def train_rbf_one_vs_one(X, y, X_test, y_test, target0, target1, C):
    X, y = prepare_one_vs_one(X, y, target0, target1)
    X_test, y_test = prepare_one_vs_one(X_test, y_test, target0, target1)
    classifer = SVC(C=C, kernel='rbf', gamma=1)
    classifer.fit(X, y)
    return classifer.score(X, y), classifer.score(X_test, y_test)

X_test, y_test = prepare_data(read_matrix('./features.test'))

C = [0.01, 1, 100, 1e4, 1e6]
for c in C:
    score = train_rbf_one_vs_one(X, y, X_test, y_test, 1, 5, c)
    print(c, score)
