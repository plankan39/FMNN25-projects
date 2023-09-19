"""
cost
----
A module containing cost functions

Types
-----
CostFunction
    type used for cost functions. 
    Alias of callable[[ArrayLike, ArrayLike], ArrayLike]

Functions
---------
quadraticLoss(a: ArrayLike, y: ArrayLike) -> ArrayLike
    calculates quadratic loss

quadraticDerivative(a: ArrayLike, y:ArrayLike) -> ArrayLike
    The derivative of quadraticLoss
"""

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

LossFunction = Callable[[ArrayLike, ArrayLike], ArrayLike]
"""
A type alias for cost function.
Alias of callable[[ArrayLike, ArrayLike], ArrayLike]
"""


def quadraticLoss(a: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    quadratic loss C = 1/2 * (y-yOut) ^ 2

    Parameters
    ----------
        a: ArrayLike - The output of the neural network
        y: ArrayLike - The desired output

    Returns
    -------
        An Array with the mean squared error
    """
    return (y - a) ** 2 / 2


def quadraticDerivative(a: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Derivative of quadraticLoss()
    d(quadraticLoss)/d(a) = a - y

    Parameters
    ----------
        a: ArrayLike - output
        y: ArrayLike - desired output
    
    Returns
    -------
        An array with the quadratic derivative applied to it.
    """

    return a - y
