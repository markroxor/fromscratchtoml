#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under The MIT License - https://opensource.org/licenses/MIT

from fromscratchtoml import np
from fromscratchtoml.neural_network.layers import Layer

from copy import deepcopy
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Dense(Layer):
    """
    This is the fully connected layer where each perceptron is connected to each perceptron of the next layer.

    Examples
    --------
    >>> from fromscratchtoml.neural_network.models import Sequential
    >>> from fromscratchtoml.neural_network.optimizers import StochasticGradientDescent
    >>> from fromscratchtoml.neural_network.layers import Dense, Activation
    >>> X1 = np.array([[0, 0],[0, 1],[1, 0], [1, 1]])
    >>> y1 = np.array([[1,0], [0,1], [0,1], [1,0]])
    >>> model = Sequential()
    >>> model.add(Dense(5, input_dim=2, seed=1))
    >>> model.add(Activation('sigmoid'))
    >>> model.add(Dense(5, seed=2))
    >>> model.add(Activation('sigmoid'))
    >>> model.add(Dense(2, seed=3))
    >>> sgd = StochasticGradientDescent(learning_rate=0.1)
    >>> model.compile(optimizer=sgd, loss="mean_squared_error")
    >>> model.fit(X1, y1, batch_size=4, epochs=100)
    >>> model.predict(X1, one_hot=True)
    """

    def __init__(self, units, input_dim=None, kernel_regularizer=None, seed=None):
        """
        Initialising the layer parameters.

        Parameters
        ----------
        units : int
            Number of perceptrons in a layer.
        input_dim : int, optional
            The dimensions of the input layer. Only required for the first layer.
        kernel_regularizer : fromscratchtoml.neural_network.regularizers
            The regularizer technique to used to penalise weights.
        seed : int, optional
            The seed value for mantaining reproduciblity of random weights.

        """
        if seed is not None:  # pragma: no cover
            np.random.seed(seed)

        self.units = units
        self.biases = None

        self.weights = None

        self.kernel_regularizer = kernel_regularizer

        self.w_optimizer = None
        self.b_optimizer = None

        if input_dim:
            # xavier weight initialisation, doesnt work that well with relu
            self.biases = np.random.randn(1, self.units) * np.sqrt(2. / (self.units + input_dim))
            self.weights = np.random.randn(input_dim, self.units) * np.sqrt(2. / (self.units + input_dim))

    def forward(self, X, train=False):
        """
        Forward pass the output of the previous layer by using the current layer's weights and biases.

        Parameters
        ----------
        X : numpy.ndarray
            The ouput of the previous layer
        Returns
        -------
        numpy.array : The output of the layer.
        """

        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=1)

        if self.weights is None:
            # xavier weight initialisation, doesnt work that well with relu
            self.biases = np.random.randn(1, self.units) * np.sqrt(2. / (self.units + X.shape[1]))
            self.weights = np.random.randn(X.shape[1], self.units) * np.sqrt(2. / (self.units + X.shape[1]))

        self.input = X
        self.output = np.dot(X, self.weights) + self.biases

        return self.output

    def back_propogate(self, dEdO):
        """
        Backpropogate the error, this function adds the share of dense layer to the accumulated delta.

        Parameters
        ----------
        dEdO : numpy.ndarray
            The accumulated gradient used for calculating error gradient with respect to parameters.

        Returns
        -------
        numpy.array : The accumulated gradient.
        """
        self.dEdB = np.sum(dEdO)

        dOdW = self.input
        self.dEdW = np.dot(dOdW.T, dEdO)

        dEdO = np.dot(dEdO, self.weights.T)

        return dEdO

    def optimize(self, optimizer):
        if self.w_optimizer is None:
            self.w_optimizer = deepcopy(optimizer)
            self.b_optimizer = deepcopy(optimizer)

        self.weights = self.w_optimizer.update_weights(self.weights, self.dEdW)
        self.biases = self.b_optimizer.update_weights(self.biases, self.dEdB)

        if self.kernel_regularizer is not None:
            batch_size = self.input.shape[0]
            self.weights -= self.kernel_regularizer.grad(self.weights, batch_size)
