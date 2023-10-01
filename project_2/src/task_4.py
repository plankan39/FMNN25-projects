from pprint import pprint
import numpy as np
from line_search import ExactLineSearch
from optimization import Problem
from optimization.newton import NewtonWithLineSearch
from plot_optimization import plot2dOptimization


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


if __name__ == "__main__":
    problem = Problem(rosenbrock)
    exactLineSearch = ExactLineSearch(problem.objective_function)
    optimizer = NewtonWithLineSearch(problem, exactLineSearch)

    x0 = np.array([0, -0.7])
    x_min, f_min = optimizer.optimize(x0, max_iter=100, bk=10)
    pprint(x_min)

    plot2dOptimization(problem.objective_function, x_min)
