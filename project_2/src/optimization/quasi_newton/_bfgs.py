import copy

import matplotlib.pyplot as plt
import numpy as np
from line_search import LineSearch
from scipy.optimize import minimize_scalar

from ._quasi_newton import QuasiNewtonOptimizer
from optimization import Problem


class BFGS(QuasiNewtonOptimizer):
    def __init__(
        self,
        problem: Problem,
        line_search: LineSearch,
        residual_criterion: float = 1e-10,
        cauchy_criterion: float = 0,
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
        self.line_search = line_search
        self.residual_criterion = residual_criterion
        self.cauchy_criterion = cauchy_criterion
        self.max_iterations = max_iterations if max_iterations > 0 else 100000
        self.points = []  # Store optimization path
        self.success = False  # Flag indicating whether optimization succeeded

    def optimize(self, x0):
        """
        Solve the optimization problem starting from an initial guess.

        Parameters:
        - x_0: Initial guess for the solution.

        Returns:
        - The optimized solution.
        """
        n = x.shape[0]
        xnew = x0
        gnew = self.problem.gradient_function(x0)
        Hnew = np.eye(n)
        self.points.append(copy.deepcopy(xnew))

        for _ in range(self.max_iterations):
            x = xnew
            g = gnew
            H = Hnew

            s = -Hnew @ gnew
            alpha, *_ = self.line_search.search(x, s, 0, 1e8)

            xnew = x + alpha * s
            gnew = self.problem.gradient_function(xnew)
            Hnew = self.calculate_H_bfgs(H, gnew, g, xnew, x)

            self.points.append(copy.deepcopy(xnew))
            if self.check_criterion(x, xnew, g):
                self.success = True
                break

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

    def line_search(self, x, s):
        """
        Perform line search to find an optimal step size.

        Parameters:
        - x: Current solution.
        - s: Search direction.

        Returns:
        - The optimal step size.
        """

        def func(alpha):
            return self.problem.objective_function(x + alpha * s)

        minimize_search = minimize_scalar(func)
        if minimize_search.success:
            return minimize_search.x
        else:
            raise Exception("Exact line search failed to converge.")

    def report(self):
        """
        Print a summary of the optimization results.
        """
        if self.success:
            print("Optimization Successful!")
            print("Optimal Solution:")
            print("x =", self.x)
            print("Objective Function Value =",
                  self.problem.objective_function(self.x))
            print("Number of Iterations =", len(self.points) - 1)
        else:
            print("Optimization Failed!")

    def calculate_H_bfgs(self, H, gnew, g, xnew, x):
        """
        Update the Hessian matrix using the BFGS update formula.

        Parameters:
        - H: The current H
        - gnew: current gradient.
        - g: previous gradient
        - xnew: the current point.
        - x: the previous point.

        Returns:
        - The updated H matrix.
        """

        d = xnew - x
        y = gnew-g
        d = np.reshape(d, (d.shape[0], 0))
        y = np.reshape(y, (y.shape[0], 0))

        dTy = d.T@y
        dyT = d@y.T

        Hnew = H + (1 + (y.T@H@y)/dTy) @ (d@d.T) / \
            dTy - (dyT@H + H@y@d.T)/(dTy)

        return Hnew

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
        Z = self.problem.objective_function([x, y])  # type: ignore
        levels = np.hstack(
            (np.arange(Z.min() - 1, 5, 2),
             np.arange(5, Z.max() + 1, 50))  # type: ignore
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
