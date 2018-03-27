#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Copyright (C) 2017 Dikshant Gupta <dikshant2210@gmail.com>
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
    >>> from gensim.models import TfidfModel
    >>> from gensim.corpora import Dictionary
    >>>
    >>> x = ch.Tensor([[0, 0],[0, 1],[1, 0], [1, 1]])
    >>> y = ch.Tensor([[1,0], [0,1], [0,1], [1,0]])
    >>> train_datax = ch.utils.data.TensorDataset(x, y)
    >>> nw = omg.nn.NetworkMesh([2, 5, 2])
    >>> nw.SGD(train_data=train_datax, epochs=50, batch_size=4, lr=3, test_data=train_datax)

    """
    def __init__(self, layer_architecture=None, seed=None):
        """Compute tf-idf by multiplying a local component (term frequency) with a global component
        (inverse document frequency), and normalizing the resulting documents to unit length.
        Formula for non-normalized weight of term :math:`i` in document :math:`j` in a corpus of :math:`D` documents

        .. math:: weight_{i,j} = frequency_{i,j} * log_2 \\frac{D}{document\_freq_{i}}

        or, more generally

        .. math:: weight_{i,j} = wlocal(frequency_{i,j}) * wglobal(document\_freq_{i}, D)

        so you can plug in your own custom :math:`wlocal` and :math:`wglobal` functions.

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

        Returns
        -------
        None.

        """
        ch.save(self.__dict__, file_path)
        return

    def load_model(self, file_path):
        """This function loads the saved model from a file.

        Parameters
        ----------
        file_path : str
            The path of file from where the model should be retrieved.

        Returns
        -------
        None.

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

        Returns
        -------
        None

        """

        if not hasattr(self, "layer_architecture"):
            raise NotImplementedError("Define layer architecture before calling SGD.")

        for i in range(epochs):
            batches = ch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
            for batch in batches:
                self.update_batch(batch, lr)

            logger.info("Epoch {0}:".format(i))

            if test_data:
                self.evaluate(test_data), len(test_data)

    def update_batch(self, batch, lr):
        nabla_b = [ch.zeros(biases.size()) for biases in self.layerwise_biases]
        nabla_w = [ch.zeros(weights.size()) for weights in self.layerwise_weights]

        for x, y in zip(batch[0], batch[1]):
            x = x.view(1, ch.numel(x))
            if isinstance(y, int):
                _y = ch.zeros(self.layer_architecture[-1])
                _y[y] = 1
                y = _y

            y = y.view(1, ch.numel(y))
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [ch.add(nb, dnb) for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [ch.add(nw, dnw) for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.update_weights(lr, nabla_b, nabla_w, len(batch))

    def update_weights(self, lr, nabla_b, nabla_w, batch_size):
        self.layerwise_biases = [b - (lr) * nb
                                 for b, nb in zip(self.layerwise_biases, nabla_b)]
        self.layerwise_weights = [w - (lr) * nw
                                 for w, nw in zip(self.layerwise_weights, nabla_w)]

    def backprop(self, x, y):
        nabla_b = [ch.zeros(biases.size()) for biases in self.layerwise_biases]
        nabla_w = [ch.zeros(weights.size()) for weights in self.layerwise_weights]

        activation = x
        activations = [activation]
        zs = []
        for biases, weights in zip(self.layerwise_biases, self.layerwise_weights):
            z = ch.mm(activation, weights) + biases
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * deriv_sigmoid(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = ch.mm(activations[-2].transpose(0, 1), delta)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = deriv_sigmoid(z)
            delta = ch.mm(delta, self.layerwise_weights[-l + 1].transpose(0, 1)) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = ch.mm(activations[-l - 1].transpose(0, 1), delta)
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """This function evaluates the test dataset by feed forwarding the learned
        weights across the network and calculating the number of correct evaluations
        by the network on test data.

        Parameters
        ----------
        test_data : list of (torch.Tensor, torch.Tensor) or a similar data type.
            The test data on which the results are evaluated generally after each
            epoch.

        Returns
        -------
        None

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

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
