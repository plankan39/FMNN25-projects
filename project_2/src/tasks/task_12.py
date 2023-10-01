import numpy as np
from scipy import optimize as scipy_optimize
import line_search as LineSearch
from optimization import Problem

# from optimization.quasi_newton._bfgs import BFGS
import optimizer
import chebyquad_problem


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def H_rosenbrock(x):
    dxx = 1 / (400 * x[0] ** 2 - 400 * x[1] + 2)
    dxy = x[0] / (200 * x[0] ** 2 - 200 * x[1] + 1)
    dyy = (600 * x[0] ** 2 - 200 * x[1] + 1) / (
        200 * (200 * x[0] ** 2 - 200 * x[1] + 1)
    )

    return np.array([[dxx, dxy], [dxy, dyy]])


if __name__ == "__main__":

    # starting points

    # x0 = np.array([2, -1])

    # g_tol = 1e-5
    # x_tol = 0
    # norm_ord = 2
    # max_iter = 500

    # problem = Problem(rosenbrock)

    # line_seach = LineSearch.PowellWolfeBenja(
    #     problem.objective_function, problem.gradient_function
    # )

    # opt = optimizer.CompareBFGS(
    #     problem, line_seach, max_iterations=max_iter, g_tol=g_tol, x_tol=1e-7
    # )
    # x_list = opt.optimize(x0)
    # x_min = x_list[-1]

    # print(
    #     f"xmin: {x_min} f(x_min): {problem.objective_function(x_min)} num_iterations: {len(x_list)}")

    import quadratic
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

    line_seach = LineSearch.PowellWolfeBenja(
        problem.objective_function, problem.gradient_function
    )

    opt = optimizer.CompareBFGS(
        problem, line_seach, max_iterations=max_iter, g_tol=g_tol, x_tol=1e-7,
    )
    x_list = opt.optimize(x0, analytic_H=quad.hess)
    x_min = x_list[-1]

    print(
        f"xmin: {x_min} {problem.objective_function(x_min)} num_iterations: {len(x_list)}")
    print(
        f"xmin: {x_min} analytic fmin: {quad.analytic_minimum()} ")
