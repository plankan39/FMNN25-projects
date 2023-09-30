from typing import Callable

import numpy as np

from finite_difference import finite_difference_gradient, finite_difference_hessian


class Problem:
    """
    Initialize a problem for optimization.

    Parameters:
    - objective_function: The objective function to be minimized.
    - gradient_function: (Optional) The gradient function of the objective function.
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        f_grad: Callable[[np.ndarray], np.ndarray] | None = None,
        f_hess: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.objective_function = f
        self.gradient_function = (
            f_grad
            if f_grad
            else lambda x: finite_difference_gradient(f, x, 1e-6)
        )
        self.hessian_function = (
            f_hess
            if f_hess
            else lambda x: finite_difference_hessian(x, f)
        )
