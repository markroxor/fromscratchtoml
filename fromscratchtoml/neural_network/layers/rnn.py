#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import numpy as np

from fromscratchtoml.neural_network.layers import Layer
from fromscratchtoml.neural_network.activations import Activations
import logging

import copy

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

    def __init__(self, units=100, vocab_size=None, memory_window=4, seed=None):
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

        self.xh_optim = None
        self.hh_optim = None
        self.hy_optim = None

        self.memory_window = memory_window
        self.vocab_size = vocab_size

        self.W_xh = np.random.uniform(-np.sqrt(1./vocab_size), np.sqrt(1./vocab_size), (self.vocab_size, self.units))
        self.W_hh = np.random.uniform(-np.sqrt(1./units), np.sqrt(1./units), (self.units, self.units))
        self.W_hy = np.random.uniform(-np.sqrt(1./units), np.sqrt(1./units), (self.units, self.vocab_size))

        self.W_hy = np.random.uniform(-np.sqrt(1./vocab_size), np.sqrt(1./vocab_size), (self.vocab_size, self.units))
        self.W_xh = np.random.uniform(-np.sqrt(1./units), np.sqrt(1./units), (self.units, self.vocab_size))

        np.random.seed(10)

    def to_onehot(self, X, reverse=False):
        temp = []
        for word in X:
            if reverse:
                temp.append(np.argmax(word))
            else:
                t = np.zeros(self.vocab_size)
                t[word] = 1
                temp.append(t)
        temp = np.array(temp)
        return temp

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

        # X = self.to_onehot(X)

        self.input = X

        self.time_steps, self.vocab_size = X.shape
        self.h = np.zeros((self.time_steps+1, self.units))
        self.h[-1] = np.zeros(self.units)
        self.z = np.zeros((self.time_steps+1, self.units))
        self.z[-1] = np.zeros(self.units)

        self.outputs = np.zeros_like(X)
        for t in range(self.time_steps):
            self.z[t] = np.dot(X[t], self.W_xh.T) + np.dot(self.h[t - 1], self.W_hh.T)
            self.h[t] = Activations.tanh(self.z[t])
            y_t = np.dot(self.h[t], self.W_hy.T)
            self.outputs[t] = y_t

        # self.outputs = self.to_onehot(self.outputs, reverse=True)
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
        if self.xh_optim is None:
            self.xh_optim = copy.copy(optimizer)
            self.hh_optim = copy.copy(optimizer)
            self.hy_optim = copy.copy(optimizer)


        dEdW_hy = np.zeros_like(self.W_hy)
        dEdW_hh = np.zeros_like(self.W_hh)
        dEdW_xh = np.zeros_like(self.W_xh)

        accum_grad_next = np.zeros_like(delta)
        # print("\na1", self.W_hh[0][0])
        # print(delta, "delta")

        for t in reversed(range(self.time_steps)):
            _, dhdz = Activations.tanh(self.z[t], return_deriv=True)
            # units
            dEdZ = np.dot(delta[t], self.W_hy) * dhdz
            # dEdW_hy += np.outer(self.h[t], delta[t])
            dEdW_hy += np.outer(delta[t], self.h[t])
            accum_grad_next[t] = np.dot(dEdZ, self.W_xh)
            # delta[t] *= np.dot(dhdz, self.W_xh.T)


            memory_indices = np.arange(t - self.memory_window, t+1)
            memory_indices = memory_indices[memory_indices >= 0]

            # print(list(reversed(memory_indices)), t)
            for tt in reversed(memory_indices):
                dEdW_xh += np.outer(dEdZ, self.input[tt])
                dEdW_hh += np.dot(dEdZ.T, self.h[tt-1])
                _, dhdz = Activations.tanh(self.z[tt-1], return_deriv=True)
                # dEdZ *= np.dot(self.W_hh, dhdz) 
                dEdZ = np.dot(dEdZ, self.W_hh) * dhdz

            # for tt in memory_indices:
            #     _, dhdz = Activations.tanh(self.z[tt], return_deriv=True)
            #     dEdW_xh += np.outer(self.input[tt], np.dot(self.W_hh, dhdz) * dEdZ)
            #     dEdW_hh += np.dot(self.h[tt-1], np.dot(self.W_hh, dhdz) * dEdZ)

            # dEdW_xh += np.outer(self.input[t], dEdZ)
            # dEdW_hh += np.dot(dEdZ, self.h[t-1].T).T

        # print(self.W_xh[0][0], self.W_hy[0][0], self.W_hh[0][0])
        # print(dEdW_xh[0][0], dEdW_hy[0][0], dEdW_hh[0][0], "\n")
        self.W_xh = self.xh_optim.update_weights(self.W_xh, dEdW_xh)
        self.W_hh = self.hh_optim.update_weights(self.W_hh, dEdW_hh)
        self.W_hy = self.hy_optim.update_weights(self.W_hy, dEdW_hy)

        return accum_grad_next
