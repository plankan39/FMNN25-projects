from collections.abc import Callable
from typing import Protocol

import numpy as np
import scipy


class LineSearch(Protocol):
    """Protocol class for different implementation of line search"""

    f: Callable[[np.ndarray], float]

    def search(
        self,
        x: np.ndarray,
        direction: np.ndarray,
        l_bound: float = 0,
        u_bound: float = 1e5,
    ) -> tuple[float, int]:
        """_summary_

        Args:
            x (np.ndarray): The initial point
            direction (np.ndarray): direction to search in
            l_bound (float, optional): lower bound to search. Defaults to 0.
            u_bound (float, optional): upper bound to search. Defaults to 1e5.

        Returns:
            tuple[float, int]: _description_
        """
        ...


class PowellWolfe(LineSearch):
    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        f_grad: Callable[[np.ndarray], np.ndarray],
        c1: float = 0.01,
        c2: float = 0.9,
    ) -> None:
        """Initiates attributes used to perform the lineSearch

        Args:
            f (Callable[[np.ndarray], float]): The objective function
            gradF (Callable[[np.ndarray], np.ndarray]): The gradient of f.
            c1 (float): Constant for armijo condition.
            c2 (float): Constant for wolfe condition
        """
        assert 0 < c1 < 0.5 and c1 < c2 < 1
        self.f = f
        self.f_grad = f_grad
        self.c1 = c1
        self.c2 = c2

    def armijo(self, f, x, alpha, direction, fx, gx, c1) -> bool:
        """Checks armijo condition."""
        res = f(x + alpha * direction) <= fx + c1 * alpha * gx.T @ direction
        return res

    def wolfe(self, f, f_grad, x, alpha, direction, fx, gx, c2) -> bool:
        """Checks wolfe condition."""
        res = f_grad(x + alpha * direction).T @ direction >= c2 * gx.T @ direction
        return res

    def search(
        self,
        x: np.ndarray,
        direction: np.ndarray,
        l_bound: float = 0,
        u_bound: float = 1e5,
    ) -> tuple[float, int, int]:
        fx = self.f(x)
        grad_fx = self.f_grad(x)
        fN = 1
        gN = 1

        # NOTE(benja): starting with 2 instead of 1 gets stuck on rosenbrock for some reason?!?!?!
        alpha_minus = 2
        alpha_minus /= 2
        fN += 1
        while not self.armijo(self.f, x, alpha_minus, direction, fx, grad_fx, self.c1):
            fN += 1
            alpha_minus /= 2

        alpha_plus = alpha_minus

        fN += 1
        while self.armijo(self.f, x, alpha_plus, direction, fx, grad_fx, self.c1):
            alpha_plus *= 2

        # Find a value between the bounds that fulfills the second condition
        gN += 1
        # print(alpha_minus, alpha_plus, direction)
        while not self.wolfe(
            self.f, self.f_grad, x, alpha_minus, direction, fx, grad_fx, self.c2
        ):
            alpha_0 = (alpha_plus + alpha_minus) / 2
            if self.armijo(self.f, x, alpha_0, direction, fx, grad_fx, self.c1):
                alpha_minus = alpha_0
            else:
                alpha_plus = alpha_0
            fN += 1
            gN += 1

        return alpha_minus, fN, gN


class PowellWolfeScipy(LineSearch):
    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        gradF: Callable[[np.ndarray], np.ndarray],
        c1: float = 0.01,
        c2: float = 0.9,
    ) -> None:
        assert 0 < c1 < 0.5 and c1 < c2 < 1
        self.f = f
        self.gradF = gradF
        self.c1 = c1
        self.c2 = c2

    def search(
        self,
        x: np.ndarray,
        direction: np.ndarray,
        l_bound: float = 0,
        u_bound: float = 1e5,
    ) -> tuple[float, int, int]:
        alpha, fN, gN, *_ = scipy.optimize.line_search(
            self.f, self.gradF, x, direction, c1=self.c1, c2=self.c2
        )
        return alpha, fN, gN  # type: ignore


class Identity(LineSearch):
    """Only gives 1 in in stepsize"""

    def __init__(self) -> None:
        self.f = None
        self.gradF = None

    def search(
        self,
        x: np.ndarray,
        direction: np.ndarray,
        l_bound: float = 0,
        u_bound: float = 1e5,
    ) -> tuple[float, int, int]:
        return 1, 0, 0


class ExactLineSearch(LineSearch):
    def __init__(
        self,
        f: Callable,
        epsilon: float = 1e-5,
        u_bound: float = 1e5,
    ) -> None:
        """_summary_

        Args:
            f (Callable):
            epsilon (float, optional): Defaults to 1e-5.
            u_bound (float, optional): Defaults to 1e5.
        """
        self.f = f
        self.epsilon = epsilon
        self.u_bound = u_bound

    def search(
        self,
        x: np.ndarray,
        direction: np.ndarray,
        l_bound: float = 0,
        u_bound: float = 1e5,
    ) -> tuple[float, int]:
        u_bound = self.u_bound

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
