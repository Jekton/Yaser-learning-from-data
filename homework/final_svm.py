
# import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.svm import SVC


X = np.array([
    [1, 0],
    [0, 1],
    [0, -1],
    [-1, 0],
    [0, 2],
    [0, -2],
    [-2, 0]
    ])
y = np.array([-1, -1, -1, 1, 1, 1, 1])

# Problem 11
# x1 = X[:, 0]
# x2 = X[:, 1]
# Z = np.empty((X.shape[0], 2))
# Z[:, 0] = x2**2 - 2 * x1 - 1
# Z[:, 1] = x1**2 - 2 * x2 + 1
# positives = y > 0
# plt.plot(Z[positives][:, 0], Z[positives][:, 1], 'rx')
# plt.plot(Z[~positives][:, 0], Z[~positives][:, 1], 'bo')
# plt.show()

classifer = SVC(C=sys.maxsize, kernel='poly', gamma=1, coef0=1, degree=2)
classifer.fit(X, y)
print(len(classifer.support_vectors_))
