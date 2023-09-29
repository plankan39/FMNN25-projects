import numpy as np
from . import Problem
from line_search import LineSearch
from .newton_optimizer import calc_cauchy_diff, calc_residual, NewtonOptimizer


class NewtonWithLineSearch(NewtonOptimizer):
    def __init__(self, problem: Problem, line_search: LineSearch) -> None:
        self.problem = problem
        self.line_search = line_search

    def optimize(
        self,
        x0,
        epsilon,
        max_iter,
        ak=0,
        bk=1e8,
    ):
        x_list = [x0]
        f_list = [self.problem.objective_function(x0)]
        x_new = x0
        for _ in range(max_iter):
            x = x_new
            Ginv = np.linalg.inv(self.problem.hessian_function(x))
            g = self.problem.gradient_function(x)
            s = -Ginv @ g

            gamma_min, *_ = self.line_search.search(x, s, ak, bk)
            print("gamam min", gamma_min)
            x_new = x + gamma_min * s
            f_new = self.problem.objective_function(x_new)

            x_list.append(x_new)
            f_list.append(f_new)

            residual = calc_residual(g)
            cauchy = calc_cauchy_diff(x_new, x)
            if residual < epsilon or cauchy < epsilon:
                break
        return x_list, f_list


# def minimize_newton_exact_line_search(
#     f, x0, epsilon, max_iter, ak=0, bk=1e8, line_search_epsilon=1e-4
# ):
#     x_list = [x0]
#     f_list = [f(x0)]
#     x_new = x0
#     for _ in range(max_iter):
#         x = x_new
#         Ginv = np.linalg.inv(finite_difference_hessian(x, f, h=0.1))
#         g = finite_difference_gradient(f, x, epsilon)
#         s = -Ginv @ g

#         line_search = ExactLineSearch(f)
#         # line_search = PowellWolfeScipy(
#         #     f, lambda x0: finite_difference_gradient(f, x0, epsilon)
#         # )
#         gamma_min, *_ = line_search.search(x, s, ak, bk)
#         # (gamma_min, *_) = line_search.search()
#         print("gamam min", gamma_min)
#         x_new = x + gamma_min * s
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
