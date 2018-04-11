#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import torch as ch
from omega.toolbox import sigmoid, deriv_sigmoid

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NetworkMesh(object):
    """Objects of this class form the base of neural networks.

    Examples
    --------
    >>> import omega as omg
    >>> import torch as ch
    >>> x = ch.Tensor([[0, 0],[0, 1],[1, 0], [1, 1]])
    >>> y = ch.Tensor([[1,0], [0,1], [0,1], [1,0]])
    >>> train_datax = ch.utils.data.TensorDataset(x, y)
    >>> nw = omg.nn.NetworkMesh([2, 5, 2])
    >>> nw.SGD(train_data=train_datax, epochs=50, batch_size=4, lr=3, test_data=train_datax)

    """
    def __init__(self, layer_architecture=None, seed=None):
        """Forms the network architecture defining the number of layers and the
        number of neurons in each layer of the network.

        Parameters
        ----------
        layer_architecture : a list of integers, optional
            where ith integer represents the number of neurons in the ith layer
        seed : int, optional
            a pytorch random number seed for mantaining reproducability of weights.

        """
        if layer_architecture is None:
            logger.warning(
                "Network initialized without architecture definition,\
                 Expecting the model to be loaded."
            )
            return

        if seed:
            ch.manual_seed(seed)

        self.layer_architecture = layer_architecture
        self.num_layers = len(layer_architecture)
        self.layerwise_biases = [ch.randn(1, x) for x in layer_architecture[1:]]
        self.layerwise_weights = [ch.randn(x, y) for x, y in zip(layer_architecture[:-1], layer_architecture[1:])]

    def save_model(self, file_path):
        """This function saves the model in a file for loading it in future.

        Parameters
        ----------
        file_path : str
            The path to file where the model should be saved.

        """
        ch.save(self.__dict__, file_path)
        return

    def load_model(self, file_path):
        """This function loads the saved model from a file.

        Parameters
        ----------
        file_path : str
            The path of file from where the model should be retrieved.

        """
        self.__dict__ = ch.load(file_path)
        return

    def feedforward(self, x):
        """This function propogates the input `x` across the neural network formed
        feed forwarding it across all the perceptrons.

        Parameters
        ----------
        x : torch.Tensor
            The input which is to be forwarded.

        Returns
        -------
        x : torch.Tensor
            The activation of the final output layer.

        """

        x = x.view(1, ch.numel(x))
        for biases, weights in zip(self.layerwise_biases, self.layerwise_weights):
            x = sigmoid(ch.mm(x, weights) + biases)
        return x

    def SGD(self, train_data, epochs, batch_size, lr, test_data=None):
        """This function performs stochastic gradient descent on the weights,
        for a number of epochs until we get optimum weights which return minimum
        loss.

        Parameters
        ----------
        train_data : list of (torch.Tensor, torch.Tensor) or a similar data type.
            The training data on which the weights will be optimized to yield
            minimum loss.
        epochs : int
            The number of times the network should be trained.
        batch_size : int
            The contiguous `sample` of training data after which weights should
            be updated.
        lr : float
            This is the learning rate of the network, higher learning rate implies
            higher change in weights after each updation but this might overshoot
            the weights from their optimum values.
        test_data : list of (torch.Tensor, torch.Tensor) or a similar data type
                    optional
            The test data on which the results are evaluated generally after each
            epoch.

        """

        if not hasattr(self, "layer_architecture"):
            raise NotImplementedError("Define layer architecture before calling SGD.")

        for i in range(epochs):
            batches = ch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
            for batch in batches:
                self.__update_batch(batch, lr)

            logger.info("Epoch {0}:".format(i))

            if test_data:
                self.evaluate(test_data), len(test_data)

    def __update_batch(self, batch, lr):
        """Updates the weights after iterating through the complete input batch

        Parameters
        ----------
        batch : list of (torch.Tensor, torch.Tensor) or a similar data type.
            Batch is a contiguous sample of training data after processing which,
            the weights are updated.
        lr : float
            This is the learning rate of the network, higher learning rate implies
            higher change in weights after each updation but this might overshoot
            the weights from their optimum values.

        """
        der_cost_bias = [ch.zeros(biases.size()) for biases in self.layerwise_biases]
        der_cost_weight = [ch.zeros(weights.size()) for weights in self.layerwise_weights]

        for x, y in zip(batch[0], batch[1]):
            x = x.view(1, ch.numel(x))
            if isinstance(y, int):
                _y = ch.zeros(self.layer_architecture[-1])
                _y[y] = 1
                y = _y

            y = y.view(1, ch.numel(y))
            delta_der_cost_bias, delta_der_cost_weight = self.__backprop(x, y)
            der_cost_bias = [ch.add(nb, dnb) for nb, dnb in zip(der_cost_bias, delta_der_cost_bias)]
            der_cost_weight = [ch.add(nw, dnw) for nw, dnw in zip(der_cost_weight, delta_der_cost_weight)]

        self.__update_weights(lr, der_cost_bias, der_cost_weight)

    def __update_weights(self, lr, der_cost_bias, der_cost_weight):
        """Updates the weights by following the learning rate equation.

        Parameters
        ----------
        lr : float
            This is the learning rate of the network, higher learning rate implies
            higher change in weights after each updation but this might overshoot
            the weights from their optimum values

        der_cost_bias : list of torch.Tensor matrices
            This contains the accumulated derivative of the cost function with
            respect to biases.

        der_cost_weight : list of torch.Tensor matrices
            This contains the accumulated derivative of the cost function with
            respect to weights.

        """
        self.layerwise_biases = [b - (lr) * nb
                                 for b, nb in zip(self.layerwise_biases, der_cost_bias)]
        self.layerwise_weights = [w - (lr) * nw
                                 for w, nw in zip(self.layerwise_weights, der_cost_weight)]

    def __backprop(self, x, y):
        """Backpropogates the weights and biases to find the derivative of cost function with
        respect to weights and biases, and following the chain rule of derivation .

        Parameters
        x : torch.Tensor
            The input variable on which the model is trained.
        y : torch.Tensor
            The expected output when the input is x

        Returns
        -------
        der_cost_bias : list of torch.Tensor matrices
            This contains the accumulated derivative of the cost function with
            respect to biases.

        der_cost_weight : list of torch.Tensor matrices
            This contains the accumulated derivative of the cost function with
            respect to weights.

        """
        der_cost_bias = [ch.zeros(biases.size()) for biases in self.layerwise_biases]
        der_cost_weight = [ch.zeros(weights.size()) for weights in self.layerwise_weights]

        activation = x
        activations = [activation]
        zs = []
        for biases, weights in zip(self.layerwise_biases, self.layerwise_weights):
            z = ch.mm(activation, weights) + biases
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.__cost_derivative(activations[-1], y) * deriv_sigmoid(zs[-1])
        der_cost_bias[-1] = delta
        der_cost_weight[-1] = ch.mm(activations[-2].transpose(0, 1), delta)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = deriv_sigmoid(z)
            delta = ch.mm(delta, self.layerwise_weights[-l + 1].transpose(0, 1)) * sp
            der_cost_bias[-l] = delta
            der_cost_weight[-l] = ch.mm(activations[-l - 1].transpose(0, 1), delta)
        return (der_cost_bias, der_cost_weight)

    def evaluate(self, test_data):
        """This function evaluates the test dataset by feed forwarding the learned
        weights across the network and calculating the number of correct evaluations
        by the network on test data.

        Parameters
        ----------
        test_data : list of (torch.Tensor, torch.Tensor) or a similar data type.
            The test data on which the results are evaluated generally after each
            epoch.

        """

        correct_evaluation = 0

        self.activations = []
        for d in test_data:
            X, Y = d
            X = X.view(1, ch.numel(X))
            if isinstance(Y, int):
                _y = ch.zeros(self.layer_architecture[-1])
                _y[Y] = 1
                Y = _y

            Y = Y.view(1, ch.numel(Y))
            a = self.feedforward(X)

            self.activations.append(a)

            _, prediction = ch.max(a, 1)
            _, target = ch.max(Y, 1)
            if (prediction == target).numpy():
                correct_evaluation += 1

        logger.info("Accuracy is {}".format(correct_evaluation * 100.0 / len(test_data)))
        return correct_evaluation

    def __cost_derivative(self, output_activations, y):
        """If our cost function is sum of mean of squares of errors then the
        out cost derivative function becomes the error or output_activations - y.

        Parameters
        output_activations : torch.Tensor
            The output produced by feedforwarding the input through the netowrk.
        y : torch.Tensor
            The expected output of the network

        Returns
        -------
        der_cost_bias : list of torch.Tensor matrices
            This contains the accumulated derivative of the cost function with
            respect to biases.

        der_cost_weight : list of torch.Tensor matrices
            This contains the accumulated derivative of the cost function with
            respect to weights.

        """
        return (output_activations - y)
