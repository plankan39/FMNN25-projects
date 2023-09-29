from collections.abc import Callable

import numpy as np

from ._line_search import LineSearch


class ExactLineSearch(LineSearch):
    def __init__(
        self,
        f: Callable,
        epsilon: float = 1e-4,
    ) -> None:
        """_summary_

        Args:
            f (_type_): _description_
            ak (float): The lower bound of search area
            bk (float): the upper bound of search area
            epsilon (float): The tolerance.
            ak (float, optional): _description_. Defaults to 0.
            bk (float, optional): _description_. Defaults to 1e8.
            epsilon (float, optional): _description_. Defaults to 1e-4.
        """
        self.f = f
        self.epsilon = epsilon

    def search(
        self, x: np.ndarray, direction: np.ndarray, l_bound: float, u_bound: float
    ) -> tuple[float, int]:
        """Perform exact line search

        Args:
            x (np.ndarray): The current point.
            direction (np.ndarray): The direction to perform the line search in


        Returns:
            (alpha, fN): where alpha is the step size and fN the number of times
            f was called
        """
        GOLDEN_RATIO = 0.618033988749
        fN = 0

        def phi(alpha: float) -> float:
            """
            The function f linearly parameterized as
            phi(alpha) = f(x + alpha * direction)
            """
            return self.f(x + alpha * direction)

        lb, ub = l_bound, u_bound
        while abs(ub - lb) > self.epsilon:
            sigmak = lb + (1 - GOLDEN_RATIO) * (ub - lb)
            ugmak = lb + GOLDEN_RATIO * (ub - lb)

            # Determine new interval of uncertainty based on function values at sigmak and ugmak
            # FIXME: here we have one function evaluation too much, can be optimized
            if phi(sigmak) > phi(ugmak):
                lb = sigmak
            else:
                ub = ugmak
            fN += 2

        return (ub + lb) / 2, fN
