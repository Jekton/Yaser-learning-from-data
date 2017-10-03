
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
    if w[2, 0] == 0:
        return 0
    return -(w[0, 0] + w[1, 0] * x) / w[2, 0]


def plot_line(w, style = "b-"):
    xs = [-1, 2]
    ys = [compute_y_of_line(w, x) for x in xs]
    plt.plot(xs, ys, style)


def riemann_sum(w, delta):
    s = 0
    for x in np.arange(0, 1, delta):
        y = compute_y_of_line(w, x)
        if y < 0:
            y = 0
        elif y > 1:
            y = 1
        s += y * delta
    return s


def perceptron_random(X, y, w):
    bool_y = y >= 0
    N = len(y)
    niter = 0
    while True:
        h = np.matmul(X, w) >= 0
        misclassify = (h != bool_y).flatten()

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
            w += (yi * xi).reshape((3, 1))
        niter += 1
    return w, niter


def linear_regression(X, y):
    pseudo_inv = np.linalg.inv(np.matmul(X.transpose(), X))
    pseudo_inv = np.matmul(pseudo_inv, X.transpose())
    return np.matmul(pseudo_inv, y)



def simulate(N, print_res=True):
    X = np.random.random([N, 2])
    X = np.concatenate((np.ones([N, 1]), X), axis=1)

    f = generate_random_line()
    y = np.matmul(X, f)
    greaters = (y >= 0).reshape(N)
    lesses = X[~greaters]
    greaters = X[greaters]

    y = np.sign(np.matmul(X, f))
    w = linear_regression(X, y)
    h = np.sign(np.matmul(X, w))
    error = sum(h != y) / N

    w, niter = perceptron_random(X, y, w)

    if print_res:
        plt.plot(greaters[:, 1], greaters[:, 2], 'rx')
        plt.plot(lesses[:, 1], lesses[:, 2], 'go')
        plot_line(f)
        plot_line(w, "r-")
        plt.xlim((-0.05, 1.05))
        plt.ylim((-0.05, 1.05))
        plt.show()
    return f, w, error, niter




#N = 10
#nexperiment = 10000
#delta = 0.1
#E_in = 0
#E_out = 0
#total_iters = 0
#
#for i in range(nexperiment):
#    f, g, error, niter = simulate(N, False)
#    E_in += error
#    total_iters += niter
#
#    less_of_f = riemann_sum(f, delta)
#    less_of_g = riemann_sum(g, delta)
#    error = np.abs(less_of_f - less_of_g)
#    E_out += error
#    print(i)
#print("E_in =", E_in / nexperiment)
#print("E_out =", E_out / nexperiment)
#print("average niter =", total_iters / nexperiment)



def generate_nonlinear_data(N, noise_rate, plot=False):
    X = np.random.uniform(-1, 1, (N, 2))
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = np.sign(x1 * x1 + x2 * x2 - 0.6)

    noise = np.arange(N)
    np.random.shuffle(noise)
    noise = noise[:int(N * noise_rate)]
    y[noise] = -y[noise]

    if plot:
        positives = X[y == 1]
        negatives = X[y == -1]
        plt.plot(positives[:, 0], positives[:, 1], "rx")
        plt.plot(negatives[:, 0], negatives[:, 1], "go")
        plt.xlim((-1.05, 1.05))
        plt.ylim((-1.05, 1.05))
        plt.show()

    y = y.reshape((N, 1))
    return np.concatenate((np.ones((N, 1)), X), axis=1), y




def simulate_nonlinear(N, noise_rate, delta):
    X, y = generate_nonlinear_data(N, noise_rate)
    #w = linear_regression(X, y)
    #h = np.sign(np.matmul(X, w))
    #e_in = sum(h != y) / N
    e_in = 0

    x1 = X[:, 1]
    x2 = X[:, 2]
    x1 = x1.reshape((N, 1))
    x2 = x2.reshape((N, 1))
    X_nonliear = np.concatenate((X, x1 * x2, x1 * x1, x2 * x2), axis=1)
    w = linear_regression(X_nonliear, y)

    e_out = 0
    total = 0
    def f(x1, x2):
        t = np.sign(x1 * x1 + x2 * x2 - 0.6)
        n = x1.shape[0]
        noise = np.arange(n)
        np.random.shuffle(noise)
        noise = noise[:int(n * noise_rate)]
        t[noise] = -t[noise]
        return t
    npoint = int(2 / delta)
    xs = np.linspace(-1, 1, npoint)
    ys = np.linspace(-1, 1, npoint)
    xv, yv = np.meshgrid(xs, ys)
    nrow = xv.shape[1]
    for row in range(nrow):
        x1 = xv[row, :].reshape((npoint, 1))
        x2 = yv[row, :].reshape((npoint, 1))
        target = f(x1, x2)
        X_row = np.concatenate((np.ones((npoint, 1)), x1, x2, x1 * x2, x1 * x1, x2 * x2), axis=1)
        h = np.sign(np.matmul(X_row, w))
        e_out += sum(h != target)
    total = npoint * nrow

    return e_in, w, e_out / total


N = 1000
noise_rate = 0.1
nexperiment = 10
delta = 0.001
#generate_nonlinear_data(N, 0.1, True)

E_in = 0
E_out = 0
for i in range(nexperiment):
    e_in, w, e_out = simulate_nonlinear(N, noise_rate, delta)
    E_in += e_in
    E_out += e_out
    #print("w =", w.flatten())
    print(i)

print("E_in =", E_in / nexperiment)
print("E_out =", E_out / nexperiment)
