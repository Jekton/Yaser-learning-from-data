
import matplotlib.pyplot as plt

def compute_y_of_line(w, x):
    # w0 + w1*x + w2*y = 0
    if w[2, 0] == 0:
        return 0
    return -(w[0, 0] + w[1, 0] * x) / w[2, 0]


def plot_line(w, style = "b-"):
    xs = [-1, 2]
    ys = [compute_y_of_line(w, x) for x in xs]
    plt.plot(xs, ys, style)
