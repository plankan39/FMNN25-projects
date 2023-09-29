
from matplotlib import pyplot as plt
import numpy as np


def plot2dOptimization(
    f,
    x_steps,
    x_range=(-5, 5),
    y_range=(-5, 5),
    nx=100,
    ny=100,
):
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for j in range(ny):
        for i in range(nx):
            xy = np.array([x[i], y[j]])
            Z[j, i] = f(xy)

    contour_plot = plt.contour(X, Y, Z, np.logspace(0, 3.5, 10, base=10), cmap="gray")
    plt.title("Rosenbrock Function: ")
    plt.xlabel("x")
    plt.ylabel("y")

    x_list = np.array(x_steps)

    # plt.plot(x_list, 'ro')  # ko black, ro red
    plt.plot(x_list[:, 0], x_list[:, 1], "ro")  # ko black, ro red
    plt.plot(x_list[:, 0], x_list[:, 1], "r:", linewidth=1)  # plot black dotted lines
    plt.title("Steps to find minimum")
    plt.clabel(contour_plot)

    plt.show()