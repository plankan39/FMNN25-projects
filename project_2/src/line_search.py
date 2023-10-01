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
        self.c1 = c1
        self.c2 = c2

    def search(
        self,
        x: np.ndarray,
        direction: np.ndarray,
        l_bound: float = 0,
        u_bound: float = 1e5,
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
            if fN <= 10:
                print(f"phi(0) = {gradFX}")
                print(f"phi(a) = {phi_prime}")
                print(f"{np.abs(phi_prime)} <= {-self.c2 * direction.T.dot(gradFX)}")
                print(f"{np.abs(phi_prime) <= -self.c2 * direction.T.dot(gradFX)}")

            return np.abs(phi_prime) >= self.c2 * direction.T.dot(gradFX)

        # NOTE(benja): why do we do this, please explain?
        # alpha_minus = fX
        alpha_minus = 1
        print(f"lb={l_bound}, ub={u_bound}, alpha- = {alpha_minus}")
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
            print(f"alpha+={alpha_plus}")
        print(
            f"\n\n###################################\nx = {x}\ndir = {direction}\n###################################\n"
        )

        # Find a value between the bounds that fulfills the second condition
        gN += 1
        while not wolfe(alpha_minus):
            alpha_0 = (alpha_plus + alpha_minus) / 2
            if fN <= 10:
                print(f"lb: {alpha_minus}, ub: {alpha_plus}, a0: {alpha_0}")
                print(f"f(a0)={self.f(x + alpha_0 * direction)}\n")
            if armijo(alpha_0):
                alpha_minus = alpha_0
            else:
                alpha_plus = alpha_0
            fN += 1
            gN += 1

        return alpha_minus, fN, gN


class PowellWolfeBenja(LineSearch):
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
            stepSizeInitial (float, optional): The initial guess for step size.
            Defaults to 2.
            c1 (float): Constant for armijo condition.
            c2 (float): Constant for wolfe condition
        """
        assert 0 < c1 < 0.5 and c1 < c2 < 1
        self.f = f
        self.f_grad = f_grad
        self.c1 = c1
        self.c2 = c2

    def armijo(self, f, x, alpha, direction, fx, gx, c1) -> bool:
        res = f(x + alpha * direction) <= fx + \
            c1*alpha*gx.T @ direction
        return res

    def wolfe(self, f, f_grad, x, alpha, direction, fx, gx, c2) -> bool:
        res = f_grad(x + alpha*direction).T @ direction >= c2*gx.T@direction
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
        while not self.wolfe(self.f, self.f_grad, x, alpha_minus, direction, fx, grad_fx, self.c2):
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
        self.c1 = c1
        self.c2 = c2

    def search(
        self,
        x: np.ndarray,
        direction: np.ndarray,
        l_bound: float = 0,
        u_bound: float = 1e5,
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

        alpha, fN, gN, *_ = scipy.optimize.line_search(
            self.f, self.gradF, x, direction, c1=self.c1, c2=self.c2
        )
        return alpha, fN, gN  # type: ignore


class Identity(LineSearch):
    def __init__(self) -> None:
        """Initiates attributes used to perform the lineSearch

        Args:
            f (Callable[[np.ndarray], float]): The objective function
            gradF (Callable[[np.ndarray], np.ndarray]): The gradient of f.
            stepSizeInitial (float, optional): The initial guess for step size.
            Defaults to 2.
            c1 (float): Constant for armijo condition.
            c2 (float): Constant for wolfe condition
        """
        self.f = None
        self.gradF = None

    def search(
        self,
        x: np.ndarray,
        direction: np.ndarray,
        l_bound: float = 0,
        u_bound: float = 1e5,
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
        self.u_bound = u_bound

    def search(
        self, x: np.ndarray, direction: np.ndarray, l_bound: float = 0, u_bound: float = 1e5
    ) -> tuple[float, int]:
        u_bound = self.u_bound
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
