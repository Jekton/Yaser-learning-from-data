
import matplotlib.pyplot as plt
import numpy as np


def generate_random_line():
    line = np.random.random([2, 2])
    x1 = line[0, 0]
    y1 = line[0, 1]
    x2 = line[1, 0]
    y2 = line[1, 1]
    k = (y1 - y2) / (x1 - x2)
    b = y1 - k * x1
    return b, k


m = 20

##############################################
# generate raw data and plot them
##############################################
X = np.random.random([m, 2])
X = np.concatenate((np.ones([m, 1]), X), axis=1)

line = generate_random_line()
line = np.array(line).reshape((2, 1))

plt.plot(X[:, 1], X[:, 2], "rx")
# plot the line use x = -1, 2
endpoints = np.array([[1, -1], [1, 2]])
plt.plot(endpoints[:, 1], np.matmul(endpoints, line), "b-")
plt.xlim((-0.05, 1.05))
plt.ylim((-0.05, 1.05))
plt.show()



##############################################
# classify data
##############################################
# y = kx + b => -b -kx + y = 0
w = np.empty(3)
w[0] = -line[0]
w[1] = -line[1]
w[2] = 1
w.reshape((3, 1))

y = np.matmul(X, w)
greaters = X[y >= 0]
lesses = X[y < 0]

plt.plot(greaters[:, 1], greaters[:, 2], 'rx')
plt.plot(lesses[:, 1], lesses[:, 2], 'go')
plt.plot(endpoints[:, 1], np.matmul(endpoints, line), "b-")
plt.xlim((-0.05, 1.05))
plt.ylim((-0.05, 1.05))
plt.show()
