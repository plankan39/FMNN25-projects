from pprint import pprint

import numpy as np
from line_search import ExactLineSearch
from optimizer import Optimizer, Problem
from plot_optimization import plot2dOptimization

from rosenbrock import rosenbrock


if __name__ == "__main__":
    g_tol = 1e-5
    x_tol = 0
    max_iter = 500

    problem = Problem(rosenbrock)
    line_search = ExactLineSearch(problem.objective_function, u_bound=10)
    optimizer = Optimizer(
        problem, line_search, g_tol=g_tol, x_tol=1e-5, max_iterations=max_iter
    )

    x0 = np.array([0, -0.7])
    x_list = optimizer.optimize(x0)
    x_min = x_list[-1]
    pprint(x_min)

    plot2dOptimization(problem.objective_function, x_list)
