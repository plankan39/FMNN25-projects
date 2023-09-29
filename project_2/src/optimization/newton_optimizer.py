from typing import Protocol

import numpy as np

from line_search import LineSearch
from .problem import Problem


class NewtonOptimizer(Protocol):
    problem: Problem

    def optimize(self, *args):
        ...


class QuasiNewtonOptimizer(Protocol):
    problem: Problem
    lineSearch: LineSearch

    def optimize(self, *args):
        ...


def calc_residual(gk):
    return np.linalg.norm(gk)


def calc_cauchy_diff(x_k_1, x_k):
    return np.linalg.norm(x_k_1 - x_k)
