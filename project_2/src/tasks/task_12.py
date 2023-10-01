import quadratic
import numpy as np
import line_search as LineSearch
from optimizer import Problem
from rosenbrock import rosenbrock

# from optimization.quasi_newton._bfgs import BFGS
import optimizer


if __name__ == "__main__":
    n = 100
    Q, q = quadratic.positive_definite_quadratic_data(n)
    quad = quadratic.f_quadratic(Q, q, n)
    problem = Problem(quad.eval)
    # problem = Problem(quad.eval, quad.grad, quad.hess)

    x0 = np.random.rand(n)

    g_tol = 1e-5
    x_tol = 0
    norm_ord = 2
    max_iter = 500

    line_seach = LineSearch.PowellWolfe(
        problem.objective_function, problem.gradient_function
    )

    opt = optimizer.CompareBFGS(
        problem,
        line_seach,
        max_iterations=max_iter,
        g_tol=g_tol,
        x_tol=1e-7,
    )
    x_list = opt.optimize(x0, analytic_H=quad.hess)
    x_min = x_list[-1]

    print(
        f"xmin: {x_min} {problem.objective_function(x_min)} num_iterations: {len(x_list)}"
    )
    print(f"xmin: {x_min} analytic fmin: {quad.analytic_minimum()} ")
