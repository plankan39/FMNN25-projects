from typing import Protocol

from ..problem import Problem
from line_search import LineSearch


class QuasiNewtonOptimizer(Protocol):
    problem: Problem
    lineSearch: LineSearch

    def optimize(self, *args):
        ...
