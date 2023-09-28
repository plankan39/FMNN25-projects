import numpy as np


def parametrize_function(f, gradient, point):
    """
    Returns a function 'g' such that g(x) gives the value of the function 'f' at a point determined by a step 'x' along the 'gradient' from 'point'.
    """
    gradient = np.array(gradient)
    point = np.array(point)
    return lambda x: f(*(point + gradient * x))


def exact_line_search_2d(f, ak, bk, alpha, epsilon):
    """
    Performs the exact line search for a function 'f' using the golden section method.
    f: The function to be minimized.
    ak, bk: The initial interval.
    alpha: The golden ratio constant.
    epsilon: The tolerance.
    Returns the x value where the function 'f' has a minimum in the interval [ak, bk].
    """
    # Iteratively reduce the interval [ak, bk] until its width is less than epsilon
    while abs(bk - ak) > epsilon:
        print(f"Interval: [{ak}, {bk}]")

        # Using golden section search
        sigmak = ak + (1 - alpha) * (bk - ak)
        ugmak = ak + alpha * (bk - ak)

        # Determine new interval of uncertainty based on function values at sigmak and ugmak
        # FIXME: here we have one function evaluation too much, can be optimized
        if f(sigmak) > f(ugmak):
            ak = sigmak
        else:
            bk = ugmak

    return (bk + ak) / 2
