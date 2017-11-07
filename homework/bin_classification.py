
import matplotlib.pyplot as plt
import numpy as np


class LinearClassifier(object):
    def predict(self, X):
        pred = np.matmul(X, self.w)
        positives = (pred >= 0).flatten()
        pred[positives] = 1
        pred[~positives] = -1
        return pred



class BinClassSample(LinearClassifier):
    def __init__(self, N, lo=-1, hi=1):
        self.N = N
        self.lo = lo
        self.hi = hi


    def generate_sample(self):
        while True:
            X = np.random.uniform(self.lo, self.hi, (self.N, 2))
            self.X = np.concatenate((np.ones((self.N, 1)), X), axis=1)
            self.w = self.__gen_line()
            y = np.matmul(self.X, self.w)
            positives = y >= 0
            y[positives] = 1
            y[~positives] = -1
            self.y = y
            # Try to generate sample with two classes
            if np.sum(positives) != self.N:
                break


    def __gen_line(self):
        line = np.random.uniform(self.lo, self.hi, (2, 2))
        x1, y1 = line[0, 0], line[0, 1]
        x2, y2 = line[1, 0], line[1, 1]
        k = (y1 - y2) / (x1 - x2)
        b = y1 - k * x1
        # y = kx + b => -b -kx + y = 0
        w = np.empty(3)
        w[0], w[1], w[2] = -b, -k, 1
        return w.reshape((3, 1))


    def plot(self):
        positives = (self.y == 1).flatten()
        pos_X = self.X[positives]
        neg_X = self.X[~positives]
        plt.plot(pos_X[:, 1], pos_X[:, 2], "rx")
        plt.plot(neg_X[:, 1], neg_X[:, 2], "go")
        self.__plot_line()
        plt.xlim((self.lo, self.hi))
        plt.ylim(plt.xlim())


    def __plot_line(self):
        import linear
        xs = [self.lo, self.hi]
        ys = [linear.compute_y_of_line(self.w, xs[0]), linear.compute_y_of_line(self.w, xs[1])]
        plt.plot(xs, ys, "r-")



class Perceptron(LinearClassifier):

    def fit(self, X, y):
        num_train = X.shape[0]
        num_feature = X.shape[1]

        w = np.zeros((num_feature, 1))
        while True:
            h = np.matmul(X, w) >= 0
            h = h.flatten()
            misclassify = h != (y >= 0)

            if not misclassify.any():
                break

            candidates = X[misclassify]
            i = np.random.randint(candidates.shape[0])
            xi = candidates[i]
            yi = y[misclassify][i]

            h = 1
            if np.matmul(xi, w) < 0:
                h = -1
            if h != yi:
                w += (yi * xi).reshape((num_feature, 1))
        self.w = w
        return self
