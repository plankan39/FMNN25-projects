from pprint import pprint

import numpy as np
from optimizer import ClassicalNewton, Problem
from plot_optimization import plot2dOptimization
from rosenbrock import rosenbrock

if __name__ == "__main__":
    g_tol = 1e-5
    x_tol = 0
    max_iter = 500

    problem = Problem(rosenbrock)

    optimizer = ClassicalNewton(
        problem, g_tol=g_tol, x_tol=1e-5, max_iterations=max_iter
    )

    x0 = np.array([0, -0.7])
    x_list = optimizer.optimize(x0)
    print(len(x_list))
    x_min = x_list[-1]
    pprint(x_list)

    plot2dOptimization(problem.objective_function, x_list)
