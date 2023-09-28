from collections.abc import Callable
from typing import Protocol

import numpy as np


class LineSearch(Protocol):
    """Protocol class for different implementation of line search"""

    f: Callable[[np.ndarray], float]

    def search(self, x: np.ndarray, direction: np.ndarray) -> tuple[float, int]:
        """Perform line search to find the best step size alpha

        Args:
            stepSizeInitial (float): the initial stepsize guess

        Returns:
            float: the optimal step size alpha
        """
        ...


class ExactLineSearch(LineSearch):
    def __init__(
        self,
        f,
        ak: float = 0,
        bk: float = 1e8,
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
        self.ak = ak
        self.bk = bk
        self.epsilon = epsilon

    def search(
        self,
        x: np.ndarray,
        direction: np.ndarray,
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

        lb, ub = self.ak, self.bk
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


class PowellWolfe(LineSearch):
    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        gradF: Callable[[np.ndarray], np.ndarray],
        stepSizeInitial: float = 2,
        c1: float = 0.01,
        c2: float = 0.9,
    ) -> None:
        """Initiates attributes used to perform the lineSearch

        Args:
            f (Callable[[np.ndarray], float]): The objective function
            gradF (Callable[[np.ndarray], np.ndarray]): The gradient of f.
            stepSizeInitial (float, optional): The initial guess for step size.
            Defaults to 2.
            c1 (float): Constant for armijo condition.
            c2 (float): Constant for wolfe condition
        """
        assert 0 < c1 < 0.5 and c1 < c2 < 1
        self.f = f
        self.gradF = gradF
        self.stepSizeInitial = stepSizeInitial
        self.c1 = c1
        self.c2 = c2

    def search(
        self,
        x: np.ndarray,
        direction: np.ndarray,
    ) -> tuple[float, int, int]:
        """Perform line search with PowellWolfe algorithm

        Args:
            x (np.ndarray): The current point
            direction (np.ndarray): A direction that is descending.


        Returns:
            (alpha: float, fN: int, gN: int): where alpha is the step size
            fN is the number of times f was called and gN is the number of times
            gradF was called.
        """
        fX = self.f(x)
        gradFX = self.gradF(x)
        fN = 1
        gN = 1

        def armijo(alpha: float) -> bool:
            """Checks the armijo condition for a specific stepsize alpha"""
            return self.f(
                x + alpha * direction
            ) <= fX + self.c1 * alpha * direction.T.dot(gradFX)

        def wolfe(alpha: float) -> bool:
            """Checks the second Powell-Wolfe condition"""
            phi_prime = direction.T.dot(self.gradF(x + alpha * direction))
            return np.abs(phi_prime) <= np.abs(self.c2 * direction.T.dot(gradFX))

        alpha_minus = self.stepSizeInitial
        alpha_plus = alpha_minus

        # Find lower and upper bound for alpha that fulfills armijo condition
        fN += 1
        while not armijo(alpha_minus):
            fN += 1
            alpha_plus = alpha_minus
            alpha_minus /= 2

        # might be worth running one gradient calculation
        # if self.wolfe(alpha_minus):
        #     return alpha_minus, self.fTimes, self.gTimes

        fN += 1
        while armijo(alpha_plus):
            alpha_plus *= 2

        # Find a value between the bounds that fulfills the second condition
        gN += 1
        while not wolfe(alpha_minus):
            alpha_0 = (alpha_plus + alpha_minus) / 2
            if armijo(alpha_0):
                alpha_minus = alpha_0
            else:
                alpha_plus = alpha_0
            fN += 1
            gN += 1

        return alpha_minus, fN, gN


if __name__ == "__main__":
    from scipy.optimize import line_search

    def f(x):
        return 0.5 * x[0] ** 2 + 4.5 * x[1] ** 2

    def gradF(x):
        return np.array((x[0], 9 * x[1]))

    x_0 = np.array([12, 110])
    d_0 = np.array([-1, -1])

    c1 = 0.01
    c2 = 0.9

    print("\nResults:")
    res = line_search(f, gradF, x_0, d_0, c1=c1, c2=c2)
    print(f"  scipy: alpha = {res[0]}, fn = {res[1]}, gn = {res[2]}")

    ls = PowellWolfe(f, gradF)
    a, fn, gn = ls.search(x_0, d_0)
    print(f"  PowellWolfe: alpha = {a}, fn = {fn}, gn = {gn}")

    print("\nCalculations:")
    print(f"  f(x) = {f(x_0)}")
    print(f"  f(x + a*d) = {f(x_0+ a * d_0)}")
    print(f"  f(x) + c1*a*d@f(x) = {f(x_0) + c1*a*d_0.T.dot(gradF(x_0))}")
    print(f"  gradF(x) = {gradF(x_0)}")
    print(f"  gradF(x + a*d) = {gradF(x_0+ a * d_0)}")
    print(f"  -d*gradF(x + ad) = {-d_0.T.dot(gradF(x_0+ a * d_0))}")
    print(f"  -c2*d*gradF(x) = {-c2*d_0.T.dot(gradF(x_0))}")
    print(f"  {f(x_0+ a * d_0)} <= {f(x_0) + c1*a*d_0.dot(gradF(x_0))}")
    print(f"  {-d_0.T.dot(gradF(x_0+ a * d_0))} <= {-c2*d_0.T.dot(gradF(x_0))}")
