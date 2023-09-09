"""
data
----
A module containing functions to load and process data.

Functions
--------

loadMNIST(mnistPickleFile: PathLike) -> tuple[tuple[np.ndarray, np.ndarray]]
    Loads the MNIST dataset from a pickle file and converts it into numpy.ndarray.
"""

from os import PathLike
import pickle
import numpy as np
from pathlib import Path


def loadMNIST(
    mnistPickleFile: PathLike = Path(__file__).parent.parent / "mnist.pkl",
) -> tuple[tuple[np.ndarray, np.ndarray]]:
    """
    Loads the MNIST dataset from a pickle file and converts it into numpy.ndarray.
    Returns a tuple containing training data, validation data and testing data.

    It has the form:

    (
        (XTrain: np.ndarray, yTrain: np.ndarray),
        (XValidation: np.ndarray, yValidation: np.ndarray),
        (XTest: np.ndarray, yTest: np.ndarray)
    )
    """

    with open(mnistPickleFile, "rb") as f:
        return (
            (np.array(X), np.array(y)) for X, y in pickle.load(f, encoding="latin1")
        )
