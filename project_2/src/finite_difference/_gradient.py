import numpy as np


def finite_difference_gradient_alt(f, p, epsilon=0.01):
    # p = np.array(p)
    gradient = np.zeros_like(p)

    for i in range(len(p)):
        p_shifted_front = p.copy()
        p_shifted_back = p.copy()

        # print("Before addition:", p_shifted_front)
        p_shifted_front[i] += epsilon
        # print("After addition:", p_shifted_front)

        p_shifted_back[i] -= epsilon

        gradient[i] = (f(*p_shifted_front) - f(*p_shifted_back)) / (2 * epsilon)

    return gradient


def finite_difference_gradient(f, x, h=0.01):
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
