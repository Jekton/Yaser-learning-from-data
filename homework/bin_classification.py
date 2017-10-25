
import matplotlib.pyplot as plt
import numpy as np


class BinClassSample:
    def __init__(self, N, lo=-1, hi=1):
        self.N = N
        self.lo = lo
        self.hi = hi


    def generate_sample(self):
        X = np.random.uniform(self.lo, self.hi, (self.N, 2))
        self.X = np.concatenate((np.ones((self.N, 1)), X), axis=1)
        self.f = self.__gen_line()
        y = np.matmul(self.X, self.f)
        positives = y >= 0
        y[positives] = 1
        y[~positives] = -1
        self.y = y


    def __gen_line(self):
        line = np.random.uniform(self.lo, self.hi, (2, 2))
        x1, y1 = line[0, 0], line[0, 1]
        x2, y2 = line[1, 0], line[1, 1]
        k = (y1 - y2) / (x1 - x2)
        b = y1 - k * x1
        # y = kx + b => -b -kx + y = 0
        f = np.empty(3)
        f[0], f[1], f[2] = -b, -k, 1
        return f.reshape((3, 1))


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
        ys = [linear.compute_y_of_line(self.f, xs[0]), linear.compute_y_of_line(self.f, xs[1])]
        plt.plot(xs, ys, "r-")




