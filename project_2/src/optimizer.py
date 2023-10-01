import numpy as np
import line_search
from line_search import LineSearch
import matplotlib.pyplot as plt
from optimization import Problem


class Optimizer:
    def __init__(
        self,
        problem: Problem,
        line_search: LineSearch,
        g_tol: float = 1e-5,
        x_tol: float = 0,
        max_iterations: int = 500,
    ):
        """
        Initialize an optimization solver.

        DEFAULT IS STANDARD NEWTON
        TODO: fix comments
        Parameters:
        - problem: The problem to be solved, an instance of the Problem class.
        - line_search: a line search method
        - residual_criterion: The termination criterion for the residual norm.
        - cauchy_criterion: The termination criterion for the Cauchy step norm.
        - max_iterations: The maximum number of iterations for the solver.
        """
        self.problem = problem
        self.line_search = line_search
        self.g_tol = g_tol
        self.x_tol = x_tol
        self.max_iterations = max_iterations
        self.success = False  # Flag indicating whether optimization succeeded

    def optimize(self, x0):
        """
        Solve the optimization problem starting from an initial guess.

        Parameters:
        - x0: Initial guess for the solution.

        Returns:
        - The optimized solution.
        """
        n = x0.shape[0]
        xnew = x0
        gnew = self.problem.gradient_function(x0)
        Hnew = np.eye(n)
        Hnew = np.linalg.inv(self.problem.hessian_function(x0))

        x_list = [xnew]

        for i in range(self.max_iterations):
            x = xnew
            g = gnew
            H = Hnew
            s = -H @ g
            alpha, *_ = self.line_search.search(x, s)

            xnew = x + alpha * s
            gnew = self.problem.gradient_function(xnew)
            Hnew = self.calculate_H(H, gnew, g, xnew, x)

            if self.check_criterion(x, xnew, g):
                self.success = True
                self.xmin = xnew
                break

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
        return (np.linalg.norm(x_next - x) < self.x_tol) or (np.linalg.norm(g) < self.g_tol)

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

    def calculate_H(self, H, gnew, g, xnew, x):
        """
        Update the Hessian matrix according to the formula specified by our problem

        Parameters:
        - H: The current approximation of the inverse Hessian
        - G: The current approximation of the Hessian
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

    def function_plot(self, min_range=(-0.5, -2), max_range=(2, 4), range_steps=(100, 100)):
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
        levels = np.hstack((np.arange(Z.min() - 1, 5, 2),
                           np.arange(5, Z.max() + 1, 50)))

        plt.figure(figsize=(8, 6))
        contour = plt.contour(x, y, Z, levels=levels, cmap='viridis')
        plt.clabel(contour, inline=1, fontsize=10, fmt='%d')

        points = np.asarray(self.points)
        plt.plot(points[:, 0], points[:, 1], marker='o',
                 color='red', linestyle='-', markersize=5)

        plt.colorbar(contour, label='Objective Function Value')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Contour Plot of Objective Function')
        plt.savefig('Contour_Plot.png')


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
        d = xnew - x
        y = gnew-g
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        Hnew = H + (d - H@y)/(d.T@H@y) @ d.T@H

        return Hnew


class BadBroyden(Optimizer):
    def calculate_H(self, H, gnew, g, xnew, x):
        d = xnew - x
        y = gnew-g
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        Hnew = H + (d - H@y)/(y.T@y) @ y.T

        return Hnew


class SymmetricBroyden(Optimizer):
    def calculate_H(self, H, gnew, g, xnew, x):
        d = xnew - x
        y = gnew-g
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        # ys_t = np.outer(y, s)
        # sy_t = np.outer(s, y)
        # denominator1 = np.dot(s, y)
        # denominator2 = np.dot(y, s - np.dot(H, y))

        # if denominator1 == 0 or denominator2 == 0:
        #     raise Exception("Denominator is zero in Symmetric Broyden update.")
        u = d - H@y
        a = 1/(u.T@y)

        Hnew = H + a@u@u.T

        return Hnew


class DFP(Optimizer):
    def calculate_H(self, H, gnew, g, xnew, x):
        d = xnew - x
        y = gnew-g
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        Hnew = H + (d@d.T)/(d.T@y) - (H@y@y.T@H)/(y.T@H@y)
        return Hnew


class BFGS(Optimizer):
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
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        dTy = d.T@y
        dyT = d@y.T

        Hnew = H + (1 + (y.T@H@y)/dTy) * (d@d.T) / \
            dTy - (dyT@H + H@y@d.T)/(dTy)

        return Hnew


if __name__ == "__main__":
    print("hello")
