from pprint import pprint

import numpy as np
from line_search import PowellWolfeScipy, PowellWolfe, PowellWolfeBenja
from optimization import Problem
from optimizer import Optimizer
from plot_optimization import plot2dOptimization


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


if __name__ == "__main__":
    g_tol = 1e-5
    x_tol = 0
    max_iter = 500

    x0 = np.array([0, -0.7])
    problem = Problem(rosenbrock)
    line_search = PowellWolfeBenja(
        problem.objective_function, problem.gradient_function
    )
    # line_search = PowellWolfeScipy(
    #     problem.objective_function, problem.gradient_function
    # )

    optimizer = Optimizer(problem, line_search, g_tol=g_tol,
                          x_tol=1e-5, max_iterations=max_iter)

    x0 = np.array([0, -0.7])
    x_list = optimizer.optimize(x0)
    x_min = x_list[-1]
    print("x_min:", x_min)
    print("f_min:", problem.objective_function(x_min))

    plot2dOptimization(problem.objective_function, x_list)
