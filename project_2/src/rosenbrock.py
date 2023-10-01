import numpy as np


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def H_rosenbrock(x):
    dxx = 1 / (400 * x[0] ** 2 - 400 * x[1] + 2)
    dxy = x[0] / (200 * x[0] ** 2 - 200 * x[1] + 1)
    dyy = (600 * x[0] ** 2 - 200 * x[1] + 1) / (
        200 * (200 * x[0] ** 2 - 200 * x[1] + 1)
    )

    return np.array([[dxx, dxy], [dxy, dyy]])
