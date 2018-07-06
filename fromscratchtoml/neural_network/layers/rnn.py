#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import numpy as np

from fromscratchtoml.neural_network.layers import Layer
from fromscratchtoml.neural_network.activations import Activations
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RNN(Layer):
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

    def __init__(self, units=100, vocab_size=None, memory_window=3, optimizer=None, seed=None):
        """
        Initialising the layer parameters.

        Parameters
        ----------
        units : int
            Number of perceptrons in a layer.
        input_dim : int, optional
            The dimensions of the input layer. Only required for the first layer.
        seed : int, optional
            The seed value for mantaining reproduciblity of random weights.

        """
        if seed is not None:
            np.random.seed(seed)

        self.units = units
        self.optimizer = optimizer
        self.memory_window = memory_window

        self.W_xh = np.random.randn(vocab_size, self.units)
        self.W_hh = np.random.randn(self.units, self.units)
        self.W_hy = np.random.randn(self.units, vocab_size)

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

        Notes
        -----
        Dimensions
        x = time_steps(words) * vocab_size
        xt = vocab_size
        W_xh = vocab_size * units
        z_t = x * W_xh = time_steps * units
        W_hh = units * units
        W_hy = units * vocab_size
        ht = units
        h = time_steps(words) * self.units
        """

        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=1)

        self.input = X

        self.time_steps, self.vocab_size = X.shape
        self.h = np.random.randn(self.time_steps, self.units)
        self.h[-1] = np.zeros(self.units)
        self.z = np.random.randn(self.time_steps, self.units)
        self.z[-1] = np.zeros(self.units)

        self.outputs = np.zeros_like(X)
        for t in range(self.time_steps):
            self.z[t] = np.dot(X[t], self.W_xh) + np.dot(self.h[t-1], self.W_hh)
            self.h[t] = Activations.tanh(self.z[t])
            y_t = np.dot(self.W_hy.T, self.h[t])
            self.outputs[t] = y_t

        if return_deriv:
            return self.outputs, 0

        return self.outputs

    def back_propogate(self, delta, optimizer):
        """
        Backpropogate the error, this function adds the share of dense layer to the accumulated delta.

        Parameters
        ----------
        delta : numpy.ndarray
            The accumulated delta used for calculating error gradient with respect to parameters.

        Returns
        -------
        numpy.array : The accumulated delta.
        numpy.array : Current updated derivative of error with respect to bias.
        numpy.array : Current updated derivative of error with respect to weight.

        Notes
        -----
        delta -> time_steps, vocab
        h     -> time_steps, units
        """

        dEdW_hy = np.zeros_like(self.W_hy)
        dEdW_hh = np.zeros_like(self.W_hh)
        dEdW_xh = np.zeros_like(self.W_xh)

        for t in reversed(range(self.time_steps)):
            dEdW_hy += np.outer(delta[t], self.h[t].T).T
            
            memory_indices = np.arange(t-self.memory_window, t+1)
            memory_indices = memory_indices[memory_indices >= 0]

            dEdW_hht = np.zeros_like(self.W_hh)
            dEdW_xht = np.zeros_like(self.W_xh)
            for tt in memory_indices:
                print(tt)
                _, dhdz = Activations.tanh(self.z[tt], return_deriv=True)

                dEdW_hht +=  self.h[tt-1]
                dEdW_hht *=  np.dot(dhdz, self.W_hh)
                dEdW_xht +=  self.input[tt]
                dEdW_xht *=  self.W_hh * dhdz
        
            dEdW_hht *= delta[t] * self.W_hh
            dEdW_xht *= delta[t] * self.W_hh
            dEdW_hh += dEdW_hht
            dEdW_xh += dEdW_xht

            delta[t] *= (Activations.tanh(self.z[t]) * self.W_xh)

        self.W_xh = optimizer.update_weights(self.W_xh, dEdW_xh)
        self.W_hh = optimizer.update_weights(self.W_hh, dEdW_hh)
        self.W_hy = optimizer.update_weights(self.W_hy, dEdW_hy)

        return delta