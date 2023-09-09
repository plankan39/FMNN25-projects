"""
activation
----------
A module containing activation functions.


Types
-----
ActivationFunction - Alias for Callable[[ArrayLike], ArrayLike]

Functions
--------
sigmoid(X: ArrayLike) -> ArrayLike
    The sigmoid function.

sigmoidDerivative(X: ArrayLike) -> ArrayLike
    The derivative of the sigmoid function.
"""
from typing import Callable
import numpy as np
from numpy.typing import ArrayLike

ActivationFunction = Callable[[ArrayLike], ArrayLike]
"""
Type describing an activation function. 
Alias for Callable[[ArrayLike], ArrayLike]
"""


def sigmoid(X: ArrayLike) -> np.ndarray:
    """
    The sigmoid function applied element wise.

    Parameters
    ----------
    X: ArrayLike
        The input array to apply the function on.

    Returns
    ----------
    An array with the same dimension with the sigmoid function applied to it
    """
    return 1 / (1 + np.exp(-X))


def sigmoidDerivative(X: ArrayLike) -> np.ndarray:
    """
    The derivative of the sigmoid function

    Parameters
    ----------
    X: ArrayLike
        The input array to apply the function on.

    Returns
    ----------
    An array with the same dimensions with the sigmoid derivative function
    applied to it
    """
    return sigmoid(X) * (1 - sigmoid(X))
