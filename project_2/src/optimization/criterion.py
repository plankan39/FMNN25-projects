import numpy as np


def calc_residual(gk):
    return np.linalg.norm(gk)


def calc_cauchy_diff(x_k_1, x_k):
    return np.linalg.norm(x_k_1 - x_k)
