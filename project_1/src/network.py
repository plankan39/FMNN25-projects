import numpy as np
from activation import ActivationFunction, sigmoid
from data import loadMNIST
from numpy.typing import ArrayLike


class Network:
    def __init__(
        self, sizes: tuple[int], activationFunction: ActivationFunction
    ) -> None:
        """
        Initializes the network with weights and biases.
        The first layer is assumed to be an input layer and will not have bias
        or weights.

        Parameters
        ----------
            sizes: Sequence[int] - A sequence where each element represents the
            number of neurons in a layer.

            activationFunction: ActivationFunction - The activation function
            used in the network.
        """
        inputSizes = sizes[:-1]
        outputSizes = sizes[1:]

        self.weights = tuple(
            np.random.randn(inp, out) for inp, out in zip(inputSizes, outputSizes)
        )
        """
        Tuple where each element represent the weights of a layer in the network. 
        
        The weights are 2d matrices where each column represents the weights of
        a node in a layer
        """

        self.biases = tuple(np.random.randn(out) for out in outputSizes)
        """
        Tuple where each element represent the biases of a layer in the network. 
        
        The biases are an array where each element represents the bias of
        a node in a layer
        """
        self.activation = activationFunction
        """The activation function used in the network"""

    @property
    def sizes(self) -> tuple[int]:
        """A tuple with the sizes of the layers in the network."""
        return tuple([self.weights[0].shape[0]] + [w.shape[1] for w in self.weights])

    def forward(self, X: ArrayLike) -> np.ndarray:
        """
        Feeds input X through the network returning the output of the network.

        Parameters
        ----------
        X: ArrayLike - Input to the network. Can be Vector or matrix.

        Returns
        ----------
        The output of the network as an numpy.ndarray
        """
        temp = X
        for weightsLayer, biasesLayer in zip(self.weights, self.biases):
            temp = self.activation(np.dot(temp, weightsLayer) + biasesLayer)
        return temp


if __name__ == "__main__":
    sizes = (3, 4, 5)
    n = Network(sizes, sigmoid)

    X = [[0.7, 0.2, 0.3], [1, 1, 1]]

    print(f"Weights:")
    for w in n.weights:
        print(f"{w}\n")

    print(f"Biases:")
    for b in n.biases:
        print(f"{b}\n")

    print(n.forward(X))
