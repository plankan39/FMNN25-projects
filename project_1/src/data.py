"""
data
----
A module containing functions to load and process data.

Functions
--------

loadMNIST(mnistPickleFile: PathLike) -> tuple[tuple[np.ndarray, np.ndarray]]
    Loads the MNIST dataset from a pickle file and converts it into numpy.ndarray.

getMiniBatches(
    X: Sequence[ArrayLike],
    y: Sequence[ArrayLike], 
    miniBatchSize: int
) -> list[tuple[Sequence[ArrayLike], Sequence[ArrayLike]]])
    Transforms A dataset into a list of mini batches.

oneHotEncode(y: ArrayLike, classesSize: int) -> ArrayLike
    One hot encode an array.
"""

import pickle
from os import PathLike
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike


def loadMNIST(
    mnistPickleFile: PathLike = Path(__file__).parent.parent / "mnist.pkl",
) -> tuple[tuple[np.ndarray, np.ndarray]]:
    """
    Loads the MNIST dataset from a pickle file and converts it into numpy.ndarray.
    Converts the wished output to oneHot encoding
    Returns a tuple containing training data, validation data and testing data.

    It has the form:

    (
        (XTrain: np.ndarray, yTrain: np.ndarray),
        (XValidation: np.ndarray, yValidation: np.ndarray),
        (XTest: np.ndarray, yTest: np.ndarray)
    )

    Parameters
    ----------
        mnistPickleFile: Pathlike - the path to the mnist dataset

    Returns
    -------
        The mnist dataset
    """

    with open(mnistPickleFile, "rb") as f:
        (xTrain, yTrain), (xValidate, yValidate), (xTest, yTest) = pickle.load(
            f, encoding="latin1"
        )

        return (
            tuple((xTrain, oneHotEncode(yTrain, 10))),
            tuple((xValidate, oneHotEncode(yValidate, 10))),
            tuple((xTest, oneHotEncode(yTest, 10))),
        )


def getMiniBatches(
    X: Sequence[ArrayLike], y: Sequence[ArrayLike], miniBatchSize: int
) -> list[tuple[Sequence[ArrayLike], Sequence[ArrayLike]]]:
    """
    Takes a dataset, randomizes and translates it into mini batches.

    Parameters
    ----------
        X: Sequence[ArrayLike] - The input part of the dataset
        y: Sequence[ArrayLike] - The output labels part of the dataset
        miniBatchSize: int - The size of each mini batch of the dataset

    Returns
    -------
        A list of mini batches of the dataset.
    """

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    XRand, yRand = (X[indices], y[indices])

    return [
        tuple((XRand[i : i + miniBatchSize], yRand[i : i + miniBatchSize]))
        for i in range(0, len(X), miniBatchSize)
    ]


def oneHotEncode(y: ArrayLike, classesSize: int) -> ArrayLike:
    """
    Encodes a 1d array into one hot encoding

    Parameters
    ----------
        y: ArrayLike - A 1d Array
        classesSize: int - The size of the classes y can take

    Returns
    -------
        An array with the one hot encoded values of y
    """
    oneHot = np.zeros((y.shape[0], classesSize))

    oneHot[np.arange(y.shape[0]), y] = 1
    return oneHot
