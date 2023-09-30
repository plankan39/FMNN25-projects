import numpy as np
from .. import Problem
from line_search import LineSearch
from ..criterion import calc_cauchy_diff, calc_residual
from ._newton import NewtonOptimizer


class NewtonWithLineSearch(NewtonOptimizer):
    def __init__(self, problem: Problem, line_search: LineSearch) -> None:
        self.problem = problem
        self.line_search = line_search

    def optimize(
        self,
        x0,
        gtol=1e-5,
        xtol=0,
        max_iter=100,
        ak=0,
        bk=1e5,
    ):
        x_list = [x0]
        f_list = [self.problem.objective_function(x0)]
        x_new = x0
        for _ in range(max_iter):
            x = x_new
            Ginv = np.linalg.inv(self.problem.hessian_function(x))
            g = self.problem.gradient_function(x)
            s = -Ginv @ g

            alpha, *_ = self.line_search.search(x, s, ak, bk)
            # print("alpha", alpha)
            x_new = x + alpha * s
            f_new = self.problem.objective_function(x_new)

            x_list.append(x_new)
            f_list.append(f_new)

            residual = calc_residual(g)
            cauchy = calc_cauchy_diff(x_new, x)
            if residual < gtol or cauchy < xtol:
                break
        return x_list, f_list
