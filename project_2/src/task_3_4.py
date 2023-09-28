from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from dataclasses import dataclass
from project_2.src.line_search.line_search import PowellWolfe, ExactLineSearch
from finite_difference import finite_difference_hessian, finite_difference_gradient


# import warnings
# warnings.filterwarnings("ignore")


@dataclass
class OptimizationProblem:
    f: Callable
    f_gradient = None
    fhessian = None


class Optimizer:
    def __init__():
        pass

    def optimize(optmization_problem: OptimizationProblem):
        ...


class ClassicalNewton(Optimizer):
    def __init__():
        pass

    def optimize(OptimizationProblem):
        ...

    ...


optimize.OptimizeResult


class OptimizationResult(dict):
    """
    Attributes
    ----------
    x_mid : ndarray
        The solution of the optimization.
    f_min: float
        The minimum value of the function
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    termination_message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nf_evaluations, nj_evaluations, nh_evaluations : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    n_iter : int
        Number of iterations performed by the optimizer.
    """

    def __repr__(self):
        order_keys = [
            "message",
            "success",
            "status",
            "fun",
            "funl",
            "x",
            "xl",
            "col_ind",
            "nit",
            "lower",
            "upper",
            "eqlin",
            "ineqlin",
            "converged",
            "flag",
            "function_calls",
            "iterations",
            "root",
        ]
        # 'slack', 'con' are redundant with residuals
        # 'crossover_nit' is probably not interesting to most users
        omit_keys = {"slack", "con", "crossover_nit"}

        def key(item):
            try:
                return order_keys.index(item[0].lower())
            except ValueError:  # item not in list
                return np.inf

        def omit_redundant(items):
            for item in items:
                if item[0] in omit_keys:
                    continue
                yield item

        def item_sorter(d):
            return sorted(omit_redundant(d.items()), key=key)

        if self.keys():
            return _dict_formatter(self, sorter=item_sorter)
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def calc_residual(gk):
    return np.linalg.norm(gk)


def calc_cauchy_diff(x_k_1, x_k):
    return np.linalg.norm(x_k_1 - x_k)


def minimize_classical_newton(f, x0, epsilon, max_iter):
    x_list = [x0]
    f_list = [f(x0)]
    x_new = x0
    for i in range(max_iter):
        x = x_new
        Ginv = np.linalg.inv(finite_difference_hessian(x, f, h=0.1))
        g = finite_difference_gradient(f, x, epsilon)
        s = -Ginv @ g  # newton direction
        x_new = x + s
        f_new = f(x_new)

        x_list.append(x_new)
        f_list.append(f_new)

        residual = calc_residual(g)
        cauchy = calc_cauchy_diff(x_new, x)
        if residual < epsilon or cauchy < epsilon:
            # print(f"residual {residual}")
            # print(f"cauchy {cauchy }")
            break

            # x_new = HESSIAN, GRADIENT)
    return x_list, f_list


def parametrize_function(f, gradient, point):
    """
    Returns a function 'g' such that g(x) gives the value of the function 'f' at a point determined by a step 'x' along the 'gradient' from 'point'.
    """
    gradient = np.array(gradient)
    point = np.array(point)
    return lambda x: f(*(point + gradient * x))


def parametrize_function_line(f, x, direction):
    """
    Returns a function 'g' such that g(x) gives the value of the function 'f' at a point determined by a step 'gamma' along the 'direction' from 'point'.
    """
    return lambda gamma: f(x + gamma * direction)


def minimize_newton_exact_line_search(
    f, x0, epsilon, max_iter, ak=0, bk=1e8, line_search_epsilon=1e-4
):
    x_list = [x0]
    f_list = [f(x0)]
    x_new = x0
    for i in range(max_iter):
        x = x_new
        Ginv = np.linalg.inv(finite_difference_hessian(x, f, h=0.1))
        g = finite_difference_gradient(f, x, epsilon)
        s = -Ginv @ g

        line_search = ExactLineSearch(f)
        gamma_min = line_search.search(x, s, ak, bk, line_search_epsilon)
        # (gamma_min, *_) = line_search.search()
        print("gamam min", gamma_min)
        x_new = x + gamma_min * s
        f_new = f(x_new)

        x_list.append(x_new)
        f_list.append(f_new)

        residual = calc_residual(g)
        cauchy = calc_cauchy_diff(x_new, x)
        if residual < epsilon or cauchy < epsilon:
            # print(f"residual {residual}")
            # print(f"cauchy {cauchy }")
            break

            # x_new = HESSIAN, GRADIENT)
    return x_list, f_list


if __name__ == "__main__":
    # TEST_quadratic_minimum()
    # TEST_hessian_on_quadratic()
    # TEST_classical_newton()
    # TEST_rosenbrock()
    # TEST_newton_exact_line_search()

    def f(x):
        return rosenbrock(x)

    # def f(x):
    #     return 0.5 * x[0] ** 2 + 4.5 * x[1] ** 2

    x0 = np.array([0, -0.7])
    # x0 = np.random.rand(2)
    # x_list, f_list = minimize_classical_newton(
    #     f, x0, epsilon=1e-6, max_iter=10)
    x_list, f_list = minimize_newton_exact_line_search(
        f, x0, epsilon=1e-6, max_iter=100, bk=10
    )
    # x_list, f_list = minimize_classical_newton(
    #     f, x0, epsilon=1e-6, max_iter=100)

    # f_min_approx = f_list[-1]
    # x_min_approx = x_list[-1]
    # x_min = np.ones_like(x0)
    # f_min = f(x_min)

    # print(f"f_min_approx: {f_list[-1]}")
    # print(x_min_approx)
    # print(f"L2(x-x_approx1): {np.linalg.norm(x_min - x_min_approx)}")

    nx = 100
    ny = 100
    # x = np.linspace(-0.5, 2, nx)
    # y = np.linspace(-2, 4, ny)
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)

    from pprint import pprint

    pprint(x_list)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for j in range(ny):
        for i in range(nx):
            xy = np.array([x[i], y[j]])
            Z[j, i] = f(xy)

    # def rosenbrockfunction(x, y): return (1-x)**2+100*(y-x**2)**2
    contour_plot = plt.contour(X, Y, Z, np.logspace(
        0, 3.5, 10, base=10), cmap="gray")
    plt.title("Rosenbrock Function: ")
    plt.xlabel("x")
    plt.ylabel("y")

    x_list = np.array(x_list)
    pprint(x_list)

    # plt.plot(x_list, 'ro')  # ko black, ro red
    plt.plot(x_list[:, 0], x_list[:, 1], "ro")  # ko black, ro red
    plt.plot(x_list[:, 0], x_list[:, 1], "r:",
             linewidth=1)  # plot black dotted lines
    plt.title("Steps to find minimum")
    plt.clabel(contour_plot)

    plt.show()
