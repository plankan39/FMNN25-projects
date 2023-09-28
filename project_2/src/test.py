import numpy as np
from scipy import optimize
from project_2.src.finite_difference import finite_difference_hessian
from project_2.src.quadratic import f_quadratic, positive_definite_quadratic_data
from project_2.src.task_3_4 import minimize_classical_newton, minimize_newton_exact_line_search, rosenbrock


def TEST_hessian_on_quadratic():
    "quick test on quadratic function to see if Hessian estimation is accurate"
    n = 10
    (Q, q) = positive_definite_quadratic_data(n, seed=False)
    f = f_quadratic(Q, q, n)
    x = np.random.rand(n)

    h = 0.1
    G = Q
    G_approx = finite_difference_hessian(x, f.eval, h)
    print(
        "L2 of difference between true hessian and approximated hessian of quadratic: \n",
        np.linalg.norm(G - G_approx, ord=2),
    )


def TEST_classical_newton():
    n = 50
    (Q, q) = positive_definite_quadratic_data(n)
    quadratic = f_quadratic(Q, q, n)

    def f(x):
        return quadratic.eval(x)

    x0 = np.random.rand(n)

    x_list, f_list = minimize_classical_newton(f, x0, epsilon=1e-6, max_iter=10)

    f_min = quadratic.analytic_minimum()
    x_min = quadratic.analytic_minimizer()
    f_min_approx = f_list[-1]
    x_min_approx = x_list[-1]

    print(f"f_min:        {f_min}")
    print(f"f_min_approx: {f_min_approx }")
    print(f"L2(x-x_approx1): {np.linalg.norm(x_min - x_min_approx)}")


def TEST_newton_exact_line_search():
    n = 50
    (Q, q) = positive_definite_quadratic_data(n)
    quadratic = f_quadratic(Q, q, n)

    def f(x):
        return quadratic.eval(x)

    x0 = np.random.rand(n)

    x_list, f_list = minimize_newton_exact_line_search(f, x0, epsilon=1e-6, max_iter=10)

    f_min = quadratic.analytic_minimum()
    x_min = quadratic.analytic_minimizer()
    f_min_approx = f_list[-1]
    x_min_approx = x_list[-1]

    print(f"f_min:        {f_min}")
    print(f"f_min_approx: {f_min_approx }")
    print(f"L2(x-x_approx1): {np.linalg.norm(x_min - x_min_approx)}")


def TEST_rosenbrock():
    worst = 0
    for _ in range(10000):
        x = np.random.rand(2)
        r1 = optimize.rosen(x)
        r2 = rosenbrock(x)
        res = np.linalg.norm(r1 - r2)
        worst = max(worst, res)

    print("worst", res)
