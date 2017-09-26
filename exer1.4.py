
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
    # y = kx + b => -b -kx + y = 0
    f = np.empty(3)
    f[0] = -b
    f[1] = -k
    f[2] = 1
    return f.reshape((3, 1))


def compute_y_of_line(w, x):
    # w0 + w1*x + w2*y = 0
    return -(w[0, 0] + w[1, 0] * x) / w[2, 0]


m = 20

##############################################
# generate raw data and plot them
##############################################
X = np.random.random([m, 2])
X = np.concatenate((np.ones([m, 1]), X), axis=1)

f = generate_random_line()

plt.plot(X[:, 1], X[:, 2], "rx")
# plot the line use x = -1, 2
xs = [-1, 2]
ys = [compute_y_of_line(f, x) for x in xs]
plt.plot(xs, ys, "b-")
plt.xlim((-0.05, 1.05))
plt.ylim((-0.05, 1.05))
plt.show()



##############################################
# classify data
##############################################

y = np.matmul(X, f)
# we must use a one dimension array to index X
greaters = (y >= 0).reshape(m)
lesses = X[~greaters]
greaters = X[greaters]

plt.plot(greaters[:, 1], greaters[:, 2], 'rx')
plt.plot(lesses[:, 1], lesses[:, 2], 'go')
plt.plot(xs, ys, "b-")
plt.xlim((-0.05, 1.05))
plt.ylim((-0.05, 1.05))
plt.show()

for i in range(m):
    if y[i] < 0:
        y[i] = -1
    else:
        y[i] = 1
print(y)



##############################################
# run a perceptron learning algorithm
##############################################
w = np.empty((3, 1))
while True:
    done = True
    for i in range(m):
        x = X[i]
        h = 1
        if np.matmul(x, w) < 0:
            h = -1
        if h != y[i]:
            w += (y[i] * x).reshape((3, 1))
            done = False
    if done:
        break


plt.plot(greaters[:, 1], greaters[:, 2], 'rx')
plt.plot(lesses[:, 1], lesses[:, 2], 'go')
plt.plot(xs, ys, "b-")
ys = [compute_y_of_line(w, x) for x in xs]
plt.plot(xs, ys, "r-")
plt.xlim((-0.05, 1.05))
plt.ylim((-0.05, 1.05))
plt.show()
