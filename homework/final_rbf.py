
import numpy as np
import sys

from sklearn.cluster import KMeans
from sklearn.svm import SVC


def target_function(x1, x2):
    return np.sign(x2 - x1 + 0.25 * np.sin(np.pi * x1))


def generate_training_data(num_data, target_function):
    X = np.random.uniform(-1, 1, (num_data, 2))
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = target_function(x1, x2)
    return X, y


def problem13():
    num_train = 100
    num_trials = 100
    success = 0
    for i in range(num_trials):
        X_train, y_train = generate_training_data(num_train, target_function)
        classifer = SVC(C=sys.maxsize, kernel='rbf', gamma=1.5)
        classifer.fit(X_train, y_train)
        if classifer.score(X_train, y_train) == 1:
            success += 1
        print(i)
    print(success / num_trials)


class RBF:

    def __init__(self, K, gamma):
        self.K = K
        self.gamma = gamma

    def fit(self, X, y):
        import linear
        kmeans = KMeans(n_clusters=self.K)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        self.centers_ = centers
        Distance = self.__compute_distance(centers, X)
        self.w_ = linear.linear_regression(Distance, y)

    def score(self, X, y):
        Distance = self.__compute_distance(self.centers_, X)
        pred = np.sign(np.matmul(Distance, self.w_)).flatten()
        return np.mean(pred == y)

    def __compute_distance(self, Centers, X):
        num_data = X.shape[0]
        num_center = Centers.shape[0]
        Distance = np.empty((num_data, num_center))
        for i in range(num_data):
            x = X[i]
            diff = Centers - x
            Distance[i] = np.sum(diff**2, axis=1).flatten()
        Distance = np.exp(-self.gamma * Distance)
        return np.hstack((Distance, np.ones((num_data, 1))))


def rbf_vs_svm(K):
    num_train = 100
    num_test = 50000
    num_matches = 1000
    num_rbf_win = 0
    X_test, y_test = generate_training_data(num_test, target_function)
    for i in range(num_matches):
        X_train, y_train = generate_training_data(num_train, target_function)
        rbf = RBF(K=K, gamma=1.5)
        rbf.fit(X_train, y_train)
        svm = SVC(C=sys.maxsize, kernel='rbf', gamma=1.5)
        svm.fit(X_train, y_train)
        rbf_score = rbf.score(X_test, y_test)
        svm_score = svm.score(X_test, y_test)
        if rbf_score > svm_score:
            num_rbf_win += 1
        print(i)
    print(num_rbf_win / num_matches)


def K9_to_12():
    num_train = 100
    num_test = 50000
    num_trials = 100
    X_test, y_test = generate_training_data(num_test, target_function)
    E_in_9 = 0
    E_out_9 = 0
    E_in_12 = 0
    E_out_12 = 0
    for i in range(num_trials):
        X_train, y_train = generate_training_data(num_train, target_function)
        k9 = RBF(K=9, gamma=1.5)
        k9.fit(X_train, y_train)
        E_in_9 += k9.score(X_train, y_train)
        E_out_9 += k9.score(X_test, y_test)
        k12 = RBF(K=12, gamma=1.5)
        k12.fit(X_train, y_train)
        E_in_12 += k12.score(X_train, y_train)
        E_out_12 += k12.score(X_test, y_test)
        print(i)
    E_in_9 /= num_trials
    E_out_9 /= num_trials
    E_in_12 /= num_trials
    E_out_12 /= num_trials
    print('E_in_9 =', E_in_9)
    print('E_out_9 =', E_out_9)
    print('E_in_12 =', E_in_12)
    print('E_out_12 =', E_out_12)


def gamma_1_5_to_2():
    num_train = 100
    num_test = 50000
    num_trials = 200
    X_test, y_test = generate_training_data(num_test, target_function)
    E_in0 = 0
    E_out0 = 0
    E_in1 = 0
    E_out1 = 0
    for i in range(num_trials):
        X_train, y_train = generate_training_data(num_train, target_function)
        rbf0 = RBF(K=9, gamma=1.5)
        rbf0.fit(X_train, y_train)
        E_in0 += rbf0.score(X_train, y_train)
        E_out0 += rbf0.score(X_test, y_test)
        rbf1 = RBF(K=9, gamma=2)
        rbf1.fit(X_train, y_train)
        E_in1 += rbf1.score(X_train, y_train)
        E_out1 += rbf1.score(X_test, y_test)
        print(i)
    E_in0 /= num_trials
    E_out0 /= num_trials
    E_in1 /= num_trials
    E_out1 /= num_trials
    print('E_in(1.5) =', E_in0)
    print('E_out(1.5) =', E_out0)
    print('E_in(2) =', E_in1)
    print('E_out(2) =', E_out1)


def E_in_0():
    num_train = 100
    num_trials = 2000
    num_E_in_0 = 0
    for i in range(num_trials):
        X_train, y_train = generate_training_data(num_train, target_function)
        rbf0 = RBF(K=9, gamma=1.5)
        rbf0.fit(X_train, y_train)
        score = rbf0.score(X_train, y_train)
        if score == 1:
            num_E_in_0 += 1
        print(i)
    print(num_E_in_0 / num_trials)
