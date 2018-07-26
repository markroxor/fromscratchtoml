#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under The MIT License - https://opensource.org/licenses/MIT

# from fromscratchtoml import np
from fromscratchtoml import np

from fromscratchtoml.neural_network.layers import Layer
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

    def __init__(self, units, input_dim=None, trainable=True, seed=None):
        """
        Initialising the layer parameters.

        Parameters
        ----------
        units : int
            Number of perceptrons in a layer.
        input_dim : int, optional
            The dimensions of the input layer. Only required for the first layer.
        trainable : bool, optional
            The weights of this layer will be updated only if this is set to true.
        seed : int, optional
            The seed value for mantaining reproduciblity of random weights.

        """
        if seed is not None:
            np.random.seed(seed)

        self.units = units
        self.biases = None
        self.weights = None
        self.trainable = trainable

        if input_dim:
            # x=1 single row
            self.biases = np.random.randn(1, self.units)
            self.weights = np.random.randn(input_dim, self.units)

    def forward(self, X, return_deriv=False):
        """
        Forward pass the output of the previous layer by using the current layer's weights and biases.

        Parameters
        ----------
        X : numpy.ndarray
            The ouput of the previous layer
        return_deriv : bool, optional
            If set to true, the function returns derivative of the output along with the output.

        Returns
        -------
        numpy.array : The output of the perceptron.
        """
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=1)

        if self.weights is None:
            self.biases = np.random.randn(1, self.units)
            self.weights = np.random.randn(X.shape[1], self.units)

        self.input = X
        self.output = np.dot(X, self.weights) + self.biases

        if return_deriv:
            return self.output, 0

        return self.output

    def back_propogate(self, dEdO):
        """
        Backpropogate the error, this function adds the share of dense layer to the accumulated delta.

        Parameters
        ----------
        delta : numpy.ndarray
            The accumulated delta used for calculating error gradient with respect to parameters.

        Returns
        -------
        numpy.array : The accumulated delta.
        """

        self.dEdB = np.sum(dEdO)
        self.dEdW = np.dot(self.input.T, dEdO)
        dEdO = np.dot(dEdO, self.weights.T)
        return dEdO

    def optimize(self, optimizer):
        self.weights = optimizer.update_weights(self.weights, self.dEdW)
        self.biases = optimizer.update_weights(self.biases, self.dEdB)
