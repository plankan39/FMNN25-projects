from typing import Protocol
import numpy as np

from ..criterion import calc_cauchy_diff, calc_residual
from ..problem import Problem


class NewtonOptimizer(Protocol):
    problem: Problem

    def optimize(self, *args):
        ...


class ClassicalNewton(NewtonOptimizer):
    def __init__(self, problem: Problem) -> None:
        self.problem = problem

    def optimize(self, x0: np.ndarray, epsilon: float = 1e-4, max_iter: int = 100):
        f = self.problem.objective_function
        gradF = self.problem.gradient_function
        hessF = self.problem.hessian_function

        x_list = [x0]
        f_list = [f(x0)]
        x_new = x0
        for _ in range(max_iter):
            x = x_new
            Ginv = np.linalg.inv(hessF(x))
            g = gradF(x)
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


# def minimize_classical_newton(f, x0, epsilon, max_iter):
#     x_list = [x0]
#     f_list = [f(x0)]
#     x_new = x0
#     for i in range(max_iter):
#         x = x_new
#         Ginv = np.linalg.inv(finite_difference_hessian(x, f, h=0.1))
#         g = finite_difference_gradient(f, x, epsilon)
#         s = -Ginv @ g  # newton direction
#         x_new = x + s
#         f_new = f(x_new)

#         x_list.append(x_new)
#         f_list.append(f_new)

#         residual = calc_residual(g)
#         cauchy = calc_cauchy_diff(x_new, x)
#         if residual < epsilon or cauchy < epsilon:
#             # print(f"residual {residual}")
#             # print(f"cauchy {cauchy }")
#             break

#             # x_new = HESSIAN, GRADIENT)
#     return x_list, f_list
