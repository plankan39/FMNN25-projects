from collections.abc import Callable
from typing import Protocol

import numpy as np


class LineSearch(Protocol):
    """Protocol class for different implementation of line search"""

    f: Callable[[np.ndarray], float]

    def search(
        self,
        x: np.ndarray,
        direction: np.ndarray,
        l_bound: float = 0,
        u_bound: float = 1e8,
    ) -> tuple[float, int]:
        """Perform line search to find the best step size alpha

        Args:
            stepSizeInitial (float): the initial stepsize guess

        Returns:
            float: the optimal step size alpha
        """
        ...
