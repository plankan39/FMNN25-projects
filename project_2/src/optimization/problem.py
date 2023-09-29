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
        objective_function: Callable[[np.ndarray], float],
        gradient_function: Callable[[np.ndarray], np.ndarray] | None = None,
        hessian_function: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.objective_function = objective_function
        self.gradient_function = (
            gradient_function
            if gradient_function
            else lambda x: finite_difference_gradient(objective_function, x, 1e-6)
        )
        self.hessian_function = (
            hessian_function
            if hessian_function
            else lambda x: finite_difference_hessian(x, objective_function)
        )
