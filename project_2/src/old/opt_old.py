# import numpy as np
# from scipy.optimize import minimize_scalar
# import copy
# import matplotlib.pyplot as plt


# class Problem:
#     def __init__(self, objective_function, gradient_function=None):
#         """
#         Initialize a problem for optimization.

#         Parameters:
#         - objective_function: The objective function to be minimized.
#         - gradient_function: (Optional) The gradient function of the objective function.
#         """
#         self.objective_function = objective_function
#         self.gradient_function = gradient_function


# class Optimization:
#     def __init__(self, problem, residual_criterion=1e-10, cauchy_criterion=1e-10, max_iterations=500):
#         """
#         Initialize an optimization solver.

#         Parameters:
#         - problem: The problem to be solved, an instance of the Problem class.
#         - residual_criterion: The termination criterion for the residual norm.
#         - cauchy_criterion: The termination criterion for the Cauchy step norm.
#         - max_iterations: The maximum number of iterations for the solver.
#         """
#         self.problem = problem
#         self.residual_criterion = residual_criterion
#         self.cauchy_criterion = cauchy_criterion
#         self.max_iterations = max_iterations if max_iterations > 0 else 100000
#         self.points = []  # Store optimization path
#         self.success = False  # Flag indicating whether optimization succeeded

#     def solve(self, x_0):
#         """
#         Solve the optimization problem starting from an initial guess.

#         Parameters:
#         - x_0: Initial guess for the solution.

#         Returns:
#         - The optimized solution.
#         """
#         x = x_next = x_0
#         H = np.eye(len(x))
#         g = self.compute_gradient(x)
#         self.points.append(copy.deepcopy(x_next))

#         for iteration in range(self.max_iterations):
#             s = -np.dot(H, g)
#             alpha = self.line_search(x, s)
#             x_next = x + alpha * s
#             self.points.append(copy.deepcopy(x_next))
#             if self.check_criterion(x, x_next, g):
#                 self.success = True
#                 break

#             g_next = self.compute_gradient(x_next)
#             H = self.calculate_hessian(H, x_next, x, g, g_next)

#             g = copy.deepcopy(g_next)
#             x = copy.deepcopy(x_next)

#         self.x = x_next
#         self.g = g_next
#         return self.x

#     def check_criterion(self, x, x_next, g):
#         """
#         Check termination criteria for the optimization.

#         Parameters:
#         - x: Current solution.
#         - x_next: Next solution.
#         - g: Gradient at the current solution.

#         Returns:
#         - True if any of the termination criteria are met, otherwise False.
#         """
#         return (np.linalg.norm(x_next - x) < self.cauchy_criterion) or (np.linalg.norm(g) < self.residual_criterion)

#     def compute_gradient(self, x):
#         """
#         Compute the gradient at a given solution.

#         Parameters:
#         - x: The solution at which the gradient is computed.

#         Returns:
#         - The gradient vector.
#         """
#         if self.problem.gradient_function:
#             return self.problem.gradient_function(x)
#         # To be implemented in derived classes
#         return self.calculate_gradient(x)

#     def calculate_gradient(self, x):
#         raise NotImplementedError

#     def line_search(self, x, s):
#         """
#         Perform line search to find an optimal step size.

#         Parameters:
#         - x: Current solution.
#         - s: Search direction.

#         Returns:
#         - The optimal step size.
#         """

#         def func(alpha):
#             return self.problem.objective_function(x + alpha * s)

#         minimize_search = minimize_scalar(func)
#         if minimize_search.success:
#             return minimize_search.x
#         else:
#             raise Exception("Exact line search failed to converge.")

#     def report(self):
#         """
#         Print a summary of the optimization results.
#         """
#         if self.success:
#             print("Optimization Successful!")
#             print("Optimal Solution:")
#             print("x =", self.x)
#             print("Objective Function Value =",
#                   self.problem.objective_function(self.x))
#             print("Number of Iterations =", len(self.points) - 1)
#         else:
#             print("Optimization Failed!")


#     def function_plot(self, min_range=(-0.5, -2), max_range=(2, 4), range_steps=(100, 100)):
#         """
#         Create a contour plot of the optimization problem's objective function.

#         Parameters:
#         - min_range: Minimum values for x and y axes.
#         - max_range: Maximum values for x and y axes.
#         - range_steps: Number of steps in x and y axes.

#         Saves the contour plot as 'Contour_Plot.png'.
#         """
#         x = np.linspace(min_range[0], max_range[0], range_steps[0])
#         y = np.linspace(min_range[1], max_range[1], range_steps[1])
#         x, y = np.meshgrid(x, y)
#         Z = self.problem.objective_function([x, y])
#         levels = np.hstack((np.arange(Z.min() - 1, 5, 2),
#                            np.arange(5, Z.max() + 1, 50)))

#         plt.figure(figsize=(8, 6))
#         contour = plt.contour(x, y, Z, levels=levels, cmap='viridis')
#         plt.clabel(contour, inline=1, fontsize=10, fmt='%d')

#         points = np.asarray(self.points)
#         plt.plot(points[:, 0], points[:, 1], marker='o',
#                  color='red', linestyle='-', markersize=5)

#         plt.colorbar(contour, label='Objective Function Value')
#         plt.xlabel('x')
#         plt.ylabel('y')
#         plt.title('Contour Plot of Objective Function')
#         plt.savefig('Contour_Plot.png')


# class BadBryden(Optimization):
#     def calculate_hessian(self, H, x_next, x, g, g_next):
#         s = x_next - x
#         y = g_next - g

#         ys_t = np.outer(y, s)
#         denominator = np.dot(s, y)

#         if denominator == 0:
#             raise Exception("Denominator is zero in Bad Broyden update.")

#         H += ys_t / denominator

#         return H


# class SymmetricBryden(Optimization):
#     def calculate_hessian(self, H, x_next, x, g, g_next):
#         s = x_next - x
#         y = g_next - g

#         ys_t = np.outer(y, s)
#         sy_t = np.outer(s, y)
#         denominator1 = np.dot(s, y)
#         denominator2 = np.dot(y, s - np.dot(H, y))

#         if denominator1 == 0 or denominator2 == 0:
#             raise Exception("Denominator is zero in Symmetric Broyden update.")

#         H += ys_t / denominator1 - np.dot(ys_t, sy_t) / denominator2

#         return H


# class DFP(Optimization):
#     def calculate_hessian(self, H, x_next, x, g, g_next):
#         # TODO
#         return H


# class BFGS(Optimization):
#     def calculate_hessian(self, H, x_next, x, g, g_next):
#         # TODO
#         return H