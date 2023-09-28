from collections.abc import Callable
from typing import Protocol

import numpy as np


class LineSearch(Protocol):
    """Protocol class for different implementation of line search"""

    f: Callable[[np.ndarray], float]
    gradF: Callable[[np.ndarray], np.ndarray]
    x: np.ndarray
    direction: np.ndarray

    def search(self, stepSizeInitial: float):
        """Perform line search to find the best step size alpha

        Args:
            stepSizeInitial (float): the initial stepsize guess

        Returns:
            float: the optimal step size alpha
        """
        ...


class PowellWolfe(LineSearch):
    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        gradF: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        direction: np.ndarray,
        c1: float,
        c2: float,
        fX: float | None = None,
        gX: np.ndarray | None = None,
    ) -> None:
        """Initiates attributes used to perform the lineSearch

        Args:
            f (Callable[[np.ndarray], float]): Objective function
            gradF (Callable[[np.ndarray], np.ndarray]): gradient of F
            x (np.ndarray): The input from where the line search is performed
            direction (np.ndarray): A descending direction
            c1 (float): parameter in armijo condition. c1<c2 and 0<c1<0.5
            c2 (float): parameter for wolfe condition. 0<c1<c2<1
            fX (float | None, optional): The value of f(x).
            Calculated if not supplied. Defaults to None.
            gX (np.ndarray | None, optional): The value of gradF(x).
            Calculated if not supplied. Defaults to None.
        """
        assert x.ndim == direction.ndim
        assert 0 < c1 and c1 < 1 / 2 and c1 < c2 and c2 < 1

        self.f = f
        self.gradF = gradF
        self.x: np.ndarray = x
        self.direction: np.ndarray = direction
        self.c1 = c1
        self.c2 = c2
        self.fX = fX if fX else f(x)
        self.gX: np.ndarray = gX if gX else gradF(x)

        # The number of times f and gradF has been called
        self.fTimes = 0 if fX else 1
        self.gTimes = 0 if gX else 1

    def phi(self, alpha: float) -> float:
        """phi(alpha) = f(x + alpha * direction)"""
        self.fTimes += 1
        return self.f(self.x + alpha * self.direction)

    def phi_prime(self, alpha: float) -> float:
        """The deravitive of phi"""
        self.gTimes += 1
        gAlpha = self.gradF(self.x + alpha * self.direction)
        return self.direction.T.dot(gAlpha)

    def armijo(self, alpha: float) -> bool:
        """Checks the armijo condition for a specific stepsize alpha"""
        return self.phi(alpha) <= self.fX + self.c1 * alpha * self.direction.T.dot(
            self.gX
        )

    def wolfe(self, alpha: float) -> bool:
        """Checks the second Powell-Wolfe condition"""
        return np.abs(self.phi_prime(alpha)) <= np.abs(
            self.c2 * self.direction.T.dot(self.gX)
        )

    def search(self, stepSizeInitial: float = 2) -> tuple[float, int, int]:
        alpha_minus = stepSizeInitial
        alpha_plus = alpha_minus

        # Find lower and upper bound for alpha that fulfills armijo condition
        while not self.armijo(alpha_minus):
            alpha_plus = alpha_minus
            alpha_minus /= 2

        # might be worth running one gradient calculation
        # if self.wolfe(alpha_minus):
        #     return alpha_minus, self.fTimes, self.gTimes

        while self.armijo(alpha_plus):
            alpha_plus *= 2

        # Find a value between the bounds that fulfills the second condition
        print(alpha_minus, alpha_plus)
        while not self.wolfe(alpha_minus):
            alpha_0 = (alpha_plus + alpha_minus) / 2
            print(alpha_0)
            if self.armijo(alpha_0):
                alpha_minus = alpha_0
            else:
                alpha_plus = alpha_0

        return alpha_minus, self.fTimes, self.gTimes


if __name__ == "__main__":
    p = OptimizationProblem(
        f=lambda x: 0.5 * x[0] ** 2 + 4.5 * x[1] ** 2,
        gradF=lambda x: np.array((x[0], 9 * x[1])),
        hessF=lambda x: x,
    )
    """
    p = OptimizationProblem(
        f=lambda x: 0.5 * x[0] ** 2,
        gradF=lambda x: np.array((x[0])),
        hessF=lambda x: x,
    )
    """

    x_0 = np.array([12, 110])
    d_0 = np.array([-1, -1])

    """
    x_0 = np.array([12])
    d_0 = np.array([-1])
    """
    c1 = 0.01
    c2 = 0.9

    ls = PowellWolfe(p.f, p.gradF, x_0, d_0, c1, c2)
    from scipy.optimize import line_search

    res = line_search(p.f, p.gradF, x_0, d_0, c1=c1, c2=c2)

    a, f, g = ls.search()

    print(res)
    print(a, f, g)

    print("\n\n")
    print(f"f(x_0) = {p.f(x_0)}")
    print(f"f(x_0 + a*d_0) = {p.f(x_0+ a * d_0)}")
    print(f"f(x) + c_1*a*d_0*f(x_0) = {p.f(x_0) + c1*a*d_0.T.dot(p.gradF(x_0))}")
    print(f"grafF(x_0) = {p.gradF(x_0)}")
    print(f"grafF(x_0 + a*d_0) = {p.gradF(x_0+ a * d_0)}")
    print(f"-d*gradF(x + ad) = {-d_0.T.dot(p.gradF(x_0+ a * d_0))}")
    print(f"-c2*d*gradF(x) = {-c2*d_0.T.dot(p.gradF(x_0))}")
