from typing import Callable

import line_search
import matplotlib.pyplot as plt
import numpy as np
from finite_difference import finite_difference_gradient, finite_difference_hessian
from line_search import LineSearch
from optimizer import Problem


class Problem:
    """
    Initialize a problem for optimization.

    Parameters:
    - f: the objective function
    - f_grad: (Optional) The gradient of he objective function.
    - f_hess: (Optional) The hessian of the objective function.
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        f_grad: Callable[[np.ndarray], np.ndarray] | None = None,
        f_hess: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.objective_function = f
        self.gradient_function = (
            f_grad if f_grad else lambda x: finite_difference_gradient(f, x, 1e-6)
        )
        self.hessian_function = (
            f_hess if f_hess else lambda x: finite_difference_hessian(x, f)
        )


class Optimizer:
    def __init__(
        self,
        problem: Problem,
        line_search: LineSearch,
        g_tol: float = 1e-5,
        x_tol: float = 0,
        max_iterations: int = 500,
        approximate_first_H: bool = False,
    ):
        """_summary_

        Args:
            problem (Problem): _description_
            line_search (LineSearch): _description_
            g_tol (float, optional): _description_. Defaults to 1e-5.
            x_tol (float, optional): _description_. Defaults to 0.
            max_iterations (int, optional): _description_. Defaults to 500.
            approximate_first_H (bool, optional): _description_. Defaults to False.
        """
        self.problem = problem
        self.line_search = line_search
        self.g_tol = g_tol
        self.x_tol = x_tol
        self.max_iterations = max_iterations
        self.approximate_first_H = approximate_first_H

    def optimize(self, x0):
        """_summary_

        Args:
            x0 (_type_): _description_

        Returns:
            _type_: _description_
        """
        n = x0.shape[0]
        xnew = x0
        gnew = self.problem.gradient_function(x0)
        # Hnew = np.eye(n)

        Hnew = np.eye(n)
        if self.approximate_first_H:
            Hnew = np.linalg.inv(self.problem.hessian_function(x0))

        x_list = [xnew]

        for _ in range(self.max_iterations):
            x = xnew
            g = gnew
            H = Hnew

            s = -H @ g
            alpha, *_ = self.line_search.search(x, s)

            xnew = x + alpha * s
            gnew = self.problem.gradient_function(xnew)
            if self.check_criterion(x, xnew, gnew):
                break
            Hnew = self.calculate_H(H, gnew, g, xnew, x)

            x_list.append(xnew)

        return x_list

    def check_criterion(self, x, x_next, g):
        """
        Check termination criteria for the optimization.

        Parameters:
        - x: Current solution.
        - x_next: Next solution.
        - g: Gradient at the current solution.

        Returns:
        - True if any of the termination criteria are met, otherwise False.
        """
        return (np.linalg.norm(x_next - x) < self.x_tol) or (
            np.linalg.norm(g) < self.g_tol
        )

    def calculate_H(self, H, gnew, g, xnew, x):
        """
        Update the Hessian matrix according to the formula specified by our problem

        Parameters:
        - H: The current approximation of the inverse Hessian
        - gnew: current gradient.
        - g: previous gradient
        - xnew: the current point.
        - x: the previous point.

        Returns:
        - The updated H and G matrices.
        """

        Gnew = self.problem.hessian_function(xnew)
        Hnew = np.linalg.inv(Gnew)

        return Hnew


class ClassicalNewton(Optimizer):
    def __init__(
        self,
        problem: Problem,
        line_search: LineSearch = line_search.Identity(),
        g_tol: float = 1e-5,
        x_tol: float = 0,
        max_iterations: int = 500,
    ):
        super().__init__(
            problem,
            line_search,
            g_tol,
            x_tol,
            max_iterations,
        )


class GoodBroyden(Optimizer):
    def calculate_H(self, H, gnew, g, xnew, x):
        d = xnew - x
        y = gnew - g
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        Hnew = H + (d - H @ y) / (d.T @ H @ y) @ d.T @ H
        return Hnew


class BadBroyden(Optimizer):
    def calculate_H(self, H, gnew, g, xnew, x):
        d = xnew - x
        y = gnew - g
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        Hnew = H + (d - H @ y) / (y.T @ y) @ y.T

        return Hnew


class SymmetricBroyden(Optimizer):
    def calculate_H(self, H, gnew, g, xnew, x):
        d = xnew - x
        y = gnew - g

        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        u = d - H @ y
        a = 1 / (u.T @ y)

        Hnew = H + a * u @ u.T

        return Hnew


class DFP(Optimizer):
    def calculate_H(self, H, gnew, g, xnew, x):
        d = xnew - x
        y = gnew - g
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        Hnew = H + (d @ d.T) / (d.T @ y) - (H @ y @ y.T @ H) / (y.T @ H @ y)
        return Hnew


class BFGS(Optimizer):
    def calculate_H(self, H, gnew, g, xnew, x):
        d = xnew - x
        y = gnew - g
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        dTy = d.T @ y
        dyT = d @ y.T

        Hnew = (
            H
            + (1 + (y.T @ H @ y) / dTy) * (d @ d.T) / dTy
            - (dyT @ H + H @ y @ d.T) / (dTy)
        )

        return Hnew

    def optimize(self, x0):
        x_list = [x0]
        n = x0.shape[0]
        xnew = x0
        gnew = self.problem.gradient_function(x0)
        Hnew = np.eye(n)
        # self.points.append(copy.deepcopy(xnew))

        for _ in range(self.max_iterations):
            x = xnew
            g = gnew
            H = Hnew
            # print(H)

            s = -Hnew @ gnew
            # alpha, *_ = self.line_search.search(x, s, 0, 1e8)
            alpha, *_ = self.line_search.search(x, s)

            xnew = x + alpha * s
            gnew = self.problem.gradient_function(xnew)
            Hnew = self.calculate_H(H, gnew, g, xnew, x)
            x_list.append(xnew)

            # self.points.append(copy.deepcopy(xnew))
            if self.check_criterion(x, xnew, g):
                self.success = True
                self.xmin = xnew
                break

        return x_list


class CompareBFGS(Optimizer):
    def calculate_H(self, H, gnew, g, xnew, x):
        d = xnew - x
        y = gnew - g
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        dTy = d.T @ y
        dyT = d @ y.T

        Hnew = (
            H
            + (1 + (y.T @ H @ y) / dTy) * (d @ d.T) / dTy
            - (dyT @ H + H @ y @ d.T) / (dTy)
        )

        return Hnew

    def optimize(self, x0, analytic_H):
        x_list = [x0]
        n = x0.shape[0]
        xnew = x0
        gnew = self.problem.gradient_function(x0)
        Hnew = np.eye(n)
        # self.points.append(copy.deepcopy(xnew))

        for _ in range(self.max_iterations):
            x = xnew
            g = gnew
            H = Hnew
            # print(H)

            s = -Hnew @ gnew

            # alpha, *_ = self.line_search.search(x, s, 0, 1e8)
            alpha, *_ = self.line_search.search(x, s)

            xnew = x + alpha * s
            gnew = self.problem.gradient_function(xnew)
            Hnew = self.calculate_H(H, gnew, g, xnew, x)
            Hanalytic = analytic_H(xnew)
            Hdiff = np.linalg.inv(self.problem.hessian_function(xnew))
            diff = np.linalg.norm(Hnew - Hdiff)
            print(diff)

            # Hnew = np.linalg.inv(self.problem.hessian_function(xnew))
            x_list.append(xnew)

            # self.points.append(copy.deepcopy(xnew))
            if self.check_criterion(x, xnew, g):
                self.success = True
                self.xmin = xnew
                break

        return x_list
