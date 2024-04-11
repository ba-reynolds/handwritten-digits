"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""
import os
import json
import random
from typing import List, Tuple, Any

import numpy as np

from functions import *


NetworkInput = np.ndarray[float]    # (n, 1) - determined by the number of neurons in the input layer
NetworkOutput = np.ndarray[float]   # (m, 1) - determined by the number of neurons in the output layer
GroundTruth = np.ndarray[float]     # (m, 1) - ""

TrainingData = List[Tuple[NetworkInput, GroundTruth]]
MatrixOfInputs = np.ndarray[NetworkInput]
MatrixOfTruth = np.ndarray[GroundTruth]
MiniBatch = List[Tuple[MatrixOfInputs, MatrixOfTruth]]
TestData = List[Tuple[NetworkInput, GroundTruth]]



class Network:
    def __init__(
            self,
            sizes: List[int],
            activation: NeuronActivationFunction = Sigmoid,
            cost: NetworkCostFunction = CrossEntropyCost
        ) -> None:
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method)."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.activation = activation
        self.cost = cost

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers."""
        self.biases: List[np.ndarray] = []
        for i in range(1, self.num_layers):
            rows = self.sizes[i]
            cols = 1
            self.biases.append(np.random.normal(loc=0, scale=1, size=(rows, cols)))

        self.weights: List[np.ndarray] = []
        for i in range(0, self.num_layers - 1):
            rows = self.sizes[i + 1]
            cols = self.sizes[i]
            stddev = 1 / (cols ** .5)
            self.weights.append(np.random.normal(loc=0, scale=stddev, size=(rows, cols)))

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead."""
        # The weights and biases are stored as a list of numpy matrices,
        # remember that a matrix is 2d (i.e. it has rows and columns).
        # https://youtu.be/aircAruvnKk?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&t=813

        # Note that the Network initialization code assumes that the first
        # layer of neurons is an input layer, and omits to set any biases
        # for those neurons, since biases are only ever used in computing
        # the outputs from later layers.
        self.biases: List[np.ndarray] = []
        for i in range(1, self.num_layers):
            rows = self.sizes[i]
            cols = 1
            self.biases.append(np.random.normal(loc=0, scale=1, size=(rows, cols)))

        self.weights: List[np.ndarray] = []
        for i in range(0, self.num_layers - 1):
            rows = self.sizes[i + 1]
            cols = self.sizes[i]
            self.weights.append(np.random.normal(loc=0, scale=1, size=(rows, cols)))

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """Return the output of the network if ``a`` is input."""

        # Advance through the layers of the neural network
        for b, w in zip(self.biases, self.weights):
            a = self.activation.fn(np.dot(w, a) + b)
        return a

    def SGD(
            self,
            training_data: TrainingData,
            epochs: int,
            mini_batch_size: int,
            eta: float,
            lmbda: float = 0.0,
            test_data: TestData = None,
        ) -> None:
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        n = len(training_data)       
        for j in range(epochs):
            # In each epoch, it starts by randomly shuffling the training data,
            # and then partitions it into mini-batches of the appropriate size.
            random.shuffle(training_data)
            mini_batches = []
            for k in range(0, n, mini_batch_size):
                mini_batch = training_data[k:k+mini_batch_size]
                matrix_x = np.hstack([x for x, y in mini_batch])
                matrix_y = np.hstack([y for x, y in mini_batch])
                mini_batches.append((matrix_x, matrix_y))

            # Apply a step of gradient descent for each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))

            if test_data:
                print(f"Completed epoch {j} - Evaluation {self.evaluate(test_data)}/{len(test_data)}")
            else:
                print(f"Completed epoch {j}")

    def update_mini_batch(self, mini_batch: MiniBatch, eta: float, lmbda: float, n: int) -> None:
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set."""

        # Gradient of the cost function 'C', one for the weights and the other
        # for the biases.
        # The gradient of the cost function will be the average of the gradient
        # of every training example, but since we are using stochastic gradient
        # descent we'll just get the average over a mini-batch.

        # During backpropagation, when calculating the derivatives of each
        # weight that makes up the network, we implicitly add them up over
        # all the training examples that make up the mini-batch.
        # This does not happen with the derivatives of the biases, hence
        # why ``nabla_b`` has to be added before it can be averaged by ``m``.
        matrix_x, matrix_y = mini_batch
        m = matrix_x.shape[1]

        nabla_b, nabla_w = self.backprop(matrix_x, matrix_y)
        nabla_b = [np.sum(nb, axis=1, keepdims=True) for nb in nabla_b]

        # Do a step of stochastic gradient descent
        self.weights = [(1 - eta * lmbda / n) * w - (eta / m) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]

    # TODO: fix return type
    def backprop(self, matrix_x: MatrixOfInputs, matrix_y: MatrixOfTruth) -> Any:
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function 'C_x'.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        nabla_b = [None for b in self.biases]
        nabla_w = [None for w in self.weights]

        # Feedforward
        activation = matrix_x
        activations = [matrix_x]   # activation of each layer 'a^l'
        zs = []                    # z value of each layer 'z^l'
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation.fn(z)
            activations.append(activation)

        # Calculate how much each neuron in the output layer affects C_x
        delta = self.cost.delta(zs[-1], activations[-1], matrix_y, self.activation)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * self.activation.prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data: TestData) -> int:
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        total = 0
        low = 0
        for x, y in test_data:
            result = self.feedforward(x)
            total += int(y[np.argmax(result)])
            if int(y[np.argmax(result)]) and result[np.argmax(result)] < 0.5:
                low += 1
        print(low, len(test_data)-low)
        return total


class NetworkSerializer:
    @staticmethod
    def save(network: Network, filename: str) -> None:
        """Save the neural network to the file ``filename``."""
        data = {
            "sizes": network.sizes,
            "weights": [w.tolist() for w in network.weights],
            "biases": [b.tolist() for b in network.biases],
            "cost": network.cost.__name__,
            "activation": network.activation.__name__
        }

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(filename: str) -> Network:
        """Load a neural network from the file ``filename``.  Returns an
        instance of Network."""
        with open(filename, "r") as file:
            data = json.load(file)

        import functions
        cost = getattr(functions, data["cost"])
        activation = getattr(functions, data["activation"])
        net = Network(data["sizes"], activation=activation, cost=cost)
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]

        return net
