from pprint import pprint
import numpy as np
from optimization import Problem, ClassicalNewton
from plot_optimization import plot2dOptimization


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


if __name__ == "__main__":
    problem = Problem(rosenbrock)
    optimizer = ClassicalNewton(problem)

    x0 = np.array([0, -0.7])
    x_min, f_min = optimizer.optimize(x0, epsilon=1e-6, max_iter=100)
    pprint(x_min)

    plot2dOptimization(problem.objective_function, x_min)    
        
