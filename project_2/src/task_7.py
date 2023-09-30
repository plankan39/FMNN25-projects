from pprint import pprint

import numpy as np
from line_search import PowellWolfeScipy, PowellWolfe, PowellWolfeBenja
from optimization import Problem
from optimization.newton import NewtonWithLineSearch
from plot_optimization import plot2dOptimization


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


if __name__ == "__main__":
    problem = Problem(rosenbrock)
    # powellWolfeLineSearch = PowellWolfeScipy(
    #     problem.objective_function, problem.gradient_function
    # )

    # powellWolfeLineSearch = PowellWolfe(
    #     problem.objective_function, problem.gradient_function
    # )

    powellWolfeLineSearch = PowellWolfeBenja(
        problem.objective_function, problem.gradient_function
    )
    optimizer = NewtonWithLineSearch(problem, powellWolfeLineSearch)

    x0 = np.array([0, -0.7])
    x_list, f_list = optimizer.optimize(x0, max_iter=1000, bk=10)
    x_min = x_list[-1]
    pprint(x_min)
    print("function value: ", problem.objective_function(x_min))

    plot2dOptimization(problem.objective_function, x_list)
