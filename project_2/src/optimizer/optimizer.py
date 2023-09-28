import copy
from collections.abc import Callable
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from finite_difference import finite_difference_gradient, finite_difference_hessian
from line_search.line_search import LineSearch
from scipy.optimize import minimize_scalar


def calc_residual(gk):
    return np.linalg.norm(gk)


def calc_cauchy_diff(x_k_1, x_k):
    return np.linalg.norm(x_k_1 - x_k)


class Problem:
    """
    Initialize a problem for optimization.

    Parameters:
    - objective_function: The objective function to be minimized.
    - gradient_function: (Optional) The gradient function of the objective function.
    """

    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        gradient_function: Callable[[np.ndarray], np.ndarray] | None = None,
        hessian_function: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.objective_function = objective_function
        self.gradient_function = (
            gradient_function
            if gradient_function
            else lambda x: finite_difference_gradient(objective_function, x, 1e-6)
        )
        self.hessian_function = (
            hessian_function
            if hessian_function
            else lambda x: finite_difference_hessian(x, objective_function)
        )


class NewtonOptimizer(Protocol):
    problem: Problem

    def optimize(self, *args):
        ...


class QuasiNewtonOptimizer(Protocol):
    problem: Problem
    lineSearch: LineSearch

    def optimize(self, *args):
        ...


class ClassicalNewton(NewtonOptimizer):
    def __init__(self, problem: Problem) -> None:
        self.problem = problem

    def optimize(self, x0: np.ndarray, epsilon: float = 1e-4, max_iter: int = 100):
        f = self.problem.objective_function
        gradF = self.problem.gradient_function
        hessF = self.problem.hessian_function

        x_list = [x0]
        f_list = [f(x0)]
        x_new = x0
        for _ in range(max_iter):
            x = x_new
            Ginv = np.linalg.inv(hessF(x))
            g = gradF(x)
            s = -Ginv @ g  # newton direction
            x_new = x + s
            f_new = f(x_new)

            x_list.append(x_new)
            f_list.append(f_new)

            residual = calc_residual(g)
            cauchy = calc_cauchy_diff(x_new, x)
            if residual < epsilon or cauchy < epsilon:
                # print(f"residual {residual}")
                # print(f"cauchy {cauchy }")
                break

                # x_new = HESSIAN, GRADIENT)
        return x_list, f_list


class Broyden(QuasiNewtonOptimizer):
    def __init__(
        self,
        problem: Problem,
        lineSearch: LineSearch,
        residual_criterion: float = 1e-10,
        cauchy_criterion: float = 1e-10,
        max_iterations: int = 500,
    ):
        """
        Initialize an optimization solver.

        Parameters:
        - problem: The problem to be solved, an instance of the Problem class.
        - residual_criterion: The termination criterion for the residual norm.
        - cauchy_criterion: The termination criterion for the Cauchy step norm.
        - max_iterations: The maximum number of iterations for the solver.
        """
        self.problem = problem
        self.lineSearch = lineSearch
        self.residual_criterion = residual_criterion
        self.cauchy_criterion = cauchy_criterion
        self.max_iterations = max_iterations if max_iterations > 0 else 100000
        self.points = []  # Store optimization path
        self.success = False  # Flag indicating whether optimization succeeded

    def optimize(self, x_0):
        """
        Solve the optimization problem starting from an initial guess.

        Parameters:
        - x_0: Initial guess for the solution.

        Returns:
        - The optimized solution.
        """
        x = x_next = x_0
        H = np.eye(len(x))
        g = self.problem.gradient_function(x)
        self.points.append(copy.deepcopy(x_next))

        for _ in range(self.max_iterations):
            s = -np.dot(H, g)
            alpha = self.lineSearch.search(x, s)
            x_next = x + alpha * s
            self.points.append(copy.deepcopy(x_next))
            if self.check_criterion(x, x_next, g):
                self.success = True
                break

            g_next = self.problem.gradient_function(x_next)
            H = self.calculate_hessian(H, x_next, x, g, g_next)

            g = copy.deepcopy(g_next)
            x = copy.deepcopy(x_next)

        self.x = x_next
        self.g = g_next
        return self.x

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
        return (np.linalg.norm(x_next - x) < self.cauchy_criterion) or (
            np.linalg.norm(g) < self.residual_criterion
        )


    """def line_search(self, x, s):
        " ""
        Perform line search to find an optimal step size.

        Parameters:
        - x: Current solution.
        - s: Search direction.

        Returns:
        - The optimal step size.
        " ""

        def func(alpha):
            return self.problem.objective_function(x + alpha * s)

        minimize_search = minimize_scalar(func)
        if minimize_search.success:
            return minimize_search.x
        else:
            raise Exception("Exact line search failed to converge.")
        """

    def report(self):
        """
        Print a summary of the optimization results.
        """
        if self.success:
            print("Optimization Successful!")
            print("Optimal Solution:")
            print("x =", self.x)
            print("Objective Function Value =", self.problem.objective_function(self.x))
            print("Number of Iterations =", len(self.points) - 1)
        else:
            print("Optimization Failed!")

    def calculate_hessian(self, H, x_next, x, g, g_next):
        """
        Update the Hessian matrix using the Broyden update formula.

        Parameters:
        - H: The current Hessian matrix.
        - x_next: Next solution.
        - x: Current solution.
        - g: Gradient at the current solution.
        - g_next: Gradient at the next solution.

        Returns:
        - The updated Hessian matrix.
        """
        s = x_next - x
        y = g_next - g

        u = s - np.dot(H, y)
        u_dot_y = np.dot(u, y)

        if u_dot_y == 0:
            raise Exception("Rank-one update failed. The denominator is zero.")

        H += np.outer(u, u) / u_dot_y

        return H

    def function_plot(
        self, min_range=(-0.5, -2), max_range=(2, 4), range_steps=(100, 100)
    ):
        """
        Create a contour plot of the optimization problem's objective function.

        Parameters:
        - min_range: Minimum values for x and y axes.
        - max_range: Maximum values for x and y axes.
        - range_steps: Number of steps in x and y axes.

        Saves the contour plot as 'Contour_Plot.png'.
        """
        x = np.linspace(min_range[0], max_range[0], range_steps[0])
        y = np.linspace(min_range[1], max_range[1], range_steps[1])
        x, y = np.meshgrid(x, y)
        Z = self.problem.objective_function([x, y])
        levels = np.hstack(
            (np.arange(Z.min() - 1, 5, 2), np.arange(5, Z.max() + 1, 50))
        )

        plt.figure(figsize=(8, 6))
        contour = plt.contour(x, y, Z, levels=levels, cmap="viridis")
        plt.clabel(contour, inline=1, fontsize=10, fmt="%d")

        points = np.asarray(self.points)
        plt.plot(
            points[:, 0],
            points[:, 1],
            marker="o",
            color="red",
            linestyle="-",
            markersize=5,
        )

        plt.colorbar(contour, label="Objective Function Value")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Contour Plot of Objective Function")
        plt.savefig("Contour_Plot.png")