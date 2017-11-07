

"""
It's totally wrong
"""

from sklearn import svm

from bin_classification import *


def misclassify(classifier0, classifier1, delta):
    from linear import compute_y_of_line
    total = 0
    error = 0
    X = np.ones((1, 3))
    for x1 in np.arange(-1, 1, delta):
        X[0, 1] = x1;
        for x2 in np.arange(-1, 1, delta):
            X[0, 2] = x2
            if classifier0.predict(X) != classifier1.predict(X):
                error += 1
            total += 1
    return error / total


def train(N, delta):
    sample = BinClassSample(N)
    sample.generate_sample()

    svc = svm.SVC()
    svc.fit(sample.X, sample.y.flatten())
    perceptron = Perceptron()
    perceptron.fit(sample.X, sample.y.flatten())
    return misclassify(svc, sample, delta), misclassify(perceptron, sample, delta)


N = 100
delta = 0.05
num_testing = 1000
svm_is_better = 0
error = 0
for i in range(num_testing):
    try:
        E_svm, E_pla = train(N, delta)
        if E_svm < E_pla:
            svm_is_better += 1
        print(i)
    except:
        error += 1
print("svm_is_better =", svm_is_better / num_testing)
print("error =", error)
