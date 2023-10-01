import numpy as np


def finite_difference_hessian(x, f, h=0.1):
    """
    h=0.1 actually appears to give better approximations than smaller h
    when tested on quadratics
    based on formulas from (Abramowitz and Stegun 1972) in
    https://www.sfu.ca/sasdoc/sashtml/iml/chap11/sect8.htm

    approximates the Hessian of f at x using finite differences
    """
    n = x.shape[0]

    hessian = np.zeros((n, n))

    E = np.eye(n)
    hi = h
    hj = h
    for i in range(n):
        ei = E[:, i]
        # since the hessian is symetric we only evaluate upper triangle
        for j in range(i, n):
            if i == j:
                f1 = f(x + 2 * hi * ei)
                f2 = f(x + hi * ei)
                f3 = f(x)
                f4 = f(x - hi * ei)
                f5 = f(x - 2 * hi * ei)
                df = (-f1 + 16 * f2 - 30 * f3 + 16 * f4 - f5) / (12 * hi * hi)
                hessian[i, j] = df
            else:
                ej = E[:, j]
                f1 = f(x + hi * ei + hj * ej)
                f2 = f(x + hi * ei - hj * ej)
                f3 = f(x - hi * ei + hj * ej)
                f4 = f(x - hi * ei - hj * ej)
                df = (f1 - f2 - f3 + f4) / (4 * hi * hj)
                hessian[i, j] = df
                hessian[j, i] = df
    return hessian


def finite_difference_gradient(f, x, h=1e-6):
    """
    based on formulas from (Abramowitz and Stegun 1972) in
    https://www.sfu.ca/sasdoc/sashtml/iml/chap11/sect8.htm
    """

    n = x.shape[0]

    gradient = np.zeros_like(x)
    E = np.eye(n)
    hi = h
    for i in range(n):
        ei = E[:, i]
        f1 = f(x + hi * ei)
        f2 = f(x - hi * ei)
        df = (f1 - f2) / (2 * hi)
        gradient[i] = df
    return gradient
