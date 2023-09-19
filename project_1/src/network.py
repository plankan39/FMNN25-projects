"""
data
----
A module handling Neural Networks

Classes
-------
Network - A class representing a neural network
"""

from typing import Sequence

import numpy as np
from activation import ActivationFunction, sigmoid, sigmoidDerivative
from loss import LossFunction, quadraticDerivative, quadraticLoss
from data import getMiniBatches, loadMNIST
from numpy.typing import ArrayLike
import time


class Network:
    """
    Network
    -------
    A class representing a neural network


    Initializers
    ------------
    Network(
        sizes: tuple[int],
        activationFunction: ActivationFunction,
        activationPrime: ActivationFunction,
        lossFunction: LossFunction,
        lossPrime: LossFunction
    )
        Initializes the network with the number of neurons in each layers corresponding
        to the elements in sizes.

    Methods
    -------
    forward(self, X: ArrayLike) -> np.ndarray
        Forwards input X through the network and returns the output of the network

    optimizeSGD(
        data: Sequence[tuple[ArrayLike, ArrayLike]],
        epochs: int,
        mBatchSize: int,
        lRate: float,
    )
        Optimizes the network using stochastic gradient descent

    evaluateModel(
        testData: tuple[Sequence[ArrayLike], Sequence[ArrayLike]]
    ) -> tuple[float, float]
        Evaluates the loss and the accuracy of the model

    Properties
    ----------
    sizes - the sizes of the layers in the network
    nLayers - the number of layers in the network

    """

    def __init__(
        self,
        sizes: tuple[int],
        activationFunction: ActivationFunction,
        activationPrime: ActivationFunction,
        lossFunction: LossFunction,
        lossPrime: LossFunction,
    ) -> None:
        """
        Initializes the network with weights and biases.
        The first layer is assumed to be an input layer and will not have bias
        or weights.

        Parameters
        ----------
            sizes: Sequence[int]
                A sequence where each element represents the
                number of neurons in a layer.
            activationFunction: ActivationFunction
                The activation function used in the network.
            activationPrime: ActivationFunction
                The derivative of the activation function
            lossFunction: LossFunction
                The loss function used for optimizing
            lossPrime: LossFunction
                The derivative of the loss function
        """
        inputSizes = sizes[:-1]
        outputSizes = sizes[1:]

        self.weights = tuple(
            np.random.randn(inp, out) for inp, out in zip(inputSizes, outputSizes)
        )
        """Tuple where each element represent the weights of a layer in the network."""

        self.biases = tuple(np.random.randn(1, out) for out in outputSizes)
        """
        Tuple with biases for the layers in the network.

        The biases are an array where each element represents the bias of
        a node in that layer
        """

        self.activation = activationFunction
        """The activation function used in the network"""

        self.activationPrime = activationPrime
        """The derivative of the activation function"""

        self.loss = lossFunction
        """The cost function used in optimizing the network"""

        self.lossPrime = lossPrime
        """The derivative of the cost function used"""

    @property
    def sizes(self) -> tuple[int]:
        """The sizes of the layers in the network."""
        return tuple([self.weights[0].shape[0]] + [w.shape[1] for w in self.weights])

    @property
    def nLayers(self) -> int:
        """The number of layers in the network"""
        return len(self.weights) + 1

    def optimizeSGD(
        self,
        data: Sequence[tuple[ArrayLike, ArrayLike]],
        testData: Sequence[tuple[ArrayLike, ArrayLike]],
        epochs: int,
        mBatchSize: int,
        lRate: float,
    ) -> None:
        """
        Optimizes the network with stochastic gradient descent.

        Parameters
        ----------
            data: Sequence[tuple[ArrayLike, ArrayLike]]
                The training data
            testData: Sequence[tuple[ArrayLike, ArrayLike]]
                The the testing dat validated on
            epochs: int
                the number of epochs to run
            mBatchSize: int
                the size of each mini batch
            lRate: float
                the learning rate used in SGD
        """
        for epoch in range(epochs):
            startTime = time.time()
            miniBatches = getMiniBatches(data[0], data[1], mBatchSize)
            for X, y in miniBatches:
                dwAvg = [np.zeros(w.shape) for w in self.weights]
                dbAvg = [np.zeros(b.shape) for b in self.biases]
                # Initialize the gradient vectors with zeros

                for XSample, ySample in zip(X, y):
                    mbSize = len(X)
                    dw, db = self.__backPropagate(XSample, ySample)
                    # Calculate the gradient for the sample
                    dwAvg = [
                        weight + dWeight / mbSize for weight, dWeight in zip(dwAvg, dw)
                    ]
                    dbAvg = [bias + dBias / mbSize for bias, dBias in zip(dbAvg, db)]
                    # Add the mean contribution for the sample

                self.weights = [
                    currentW - lRate * dw for currentW, dw in zip(self.weights, dwAvg)
                ]
                self.biases = [
                    currentB - lRate * db for currentB, db in zip(self.biases, dbAvg)
                ]
                # update the weights and biases

            loss, accuracy = self.evaluateModel(testData)
            # Evaluate the model
            print(
                f"######################### epoch {epoch + 1} ############################"
            )
            print(f"{time.time() - startTime} s")
            print(f"Loss: {round(loss, 4)}")
            print(f"Accuracy: {round(accuracy * 100, 4)}%")

    def forward(self, X: ArrayLike) -> np.ndarray:
        """Feeds input X through the network returning the output of the network.

        Parameters
        ----------
        X: ArrayLike - Input to the network.

        Returns
        ----------
        The output of the network
        """

        return self.__forward(X)[0][-1]

    def __forward(self, X: ArrayLike) -> tuple[tuple[np.ndarray], tuple[np.ndarray]]:
        """
        Feeds input X through the network returning the output of the network.

        Parameters
        ----------
        X: ArrayLike - Input to the network. Can be Vector or matrix.

        Returns
        ----------
        The outputs of each layer in the network and the weighted outputs of each layer
        """
        layerOutputs = [X]
        # output of each layer
        z = []
        # weighted outputs without activation function
        for weightsLayer, biasesLayer in zip(self.weights, self.biases):
            zz = layerOutputs[-1].dot(weightsLayer) + biasesLayer
            z.append(zz)
            layerOutputs.append(self.activation(zz))

        return tuple(layerOutputs), tuple(z)

    def __backPropagate(self, X: ArrayLike, y: ArrayLike):
        """
        Back propagates through the network finding the partial derivatives of
        biases and weights.

        Parameters
        ----------
            X: ArrayLike - A Sample input
            y: ArrayLike - The desired output/label

        Returns
        -------
            The partial derivatives of the weights and of the biases of the network
        """
        assert np.array(X).ndim == 1 and np.array(y).ndim == 1
        layerOutputs, z = self.__forward(X)

        db: list[np.ndarray] = [
            self.lossPrime(layerOutputs[-1], y) * self.activationPrime(z[-1])
        ]
        # partial derivative of biases d(CostFunction)/d(biases)

        dw: list[np.ndarray] = [layerOutputs[-2].T.dot(db[-1])]
        # partial derivative of weights d(CostFunction)/d(weights)

        for i in range(2, self.nLayers):
            # Back propagating through all other layers finding the partial
            # derivatives given the partial derivatives of output layer.
            d = db[0].dot(self.weights[-i + 1].T) * self.activationPrime(z[-i])
            ddw = np.array([layerOutputs[-i - 1]]).T.dot(d)
            db.insert(0, d)
            dw.insert(0, ddw)

        return dw, db

    def evaluateModel(
        self, testData: tuple[Sequence[ArrayLike], Sequence[ArrayLike]]
    ) -> tuple[float, float]:
        """
        Evaluates the model given testData

        Parameters
        ----------
            testData: tuple[Sequence[ArrayLike], Sequence[ArrayLike]]
                Test data with input and desired output

        Returns
        -------
        A tuple with (loss, accuracy) of the model
        """
        X = testData[0]
        y = testData[1]
        result = self.forward(X)

        loss = np.mean(self.loss(result, y))

        resultMax = np.argmax(result, axis=1)
        yMax = np.argmax(y, axis=1)
        # convert probability distributions to guesses

        accuracy = (resultMax == yMax).sum() / yMax.shape[0]
        # accuracy as a percentage

        return loss, accuracy


if __name__ == "__main__":
    sizes = (784, 30, 10)
    n = Network(sizes, sigmoid, sigmoidDerivative, quadraticLoss, quadraticDerivative)
    # Create a network with 3 layers with the sigmoid function as the activation function
    # and quadratic loss as the loss function

    trainingData, validationData, testData = loadMNIST()
    # loading the dataset

    epochs = 10
    miniBatchSize = 32
    learningRate = 3.0
    n.optimizeSGD(trainingData, testData, epochs, miniBatchSize, learningRate)
    # optimize the network
