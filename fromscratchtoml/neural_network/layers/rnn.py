#!/usr/bin/env python
# -*- coding: utf-8 -*-
#



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
    This is the vanilla recurrent neural network layer.

    Examples
    --------
    >>> from fromscratchtoml.neural_network.models import Sequential
    >>> from fromscratchtoml.neural_network.optimizers import StochasticGradientDescent
    >>> from fromscratchtoml.neural_network.layers import RNN, Activation
    >>> clf = Sequential()
    >>> clf.add(RNN(units=12, memory_window=10, vocab_size=61, seed=10))
    >>> clf.add(Activation('softmax'))
    >>> clf.compile(optimizer=StochasticGradientDescent(learning_rate=0.00015), loss='cross_entropy')\
    >>> clf.fit(X_train, y_train, epochs=100, batch_size=64)
    >>> clf.predict(X_test)
    """

    def __init__(self, units=100, vocab_size=None, memory_window=4, seed=None):
        """
        Initialising the layer parameters.

        Parameters
        ----------
        units : int
            Number of hidden units of the hidden states.
        vocab_size : int, optional
            The size of vocabulary.
        memory_window: The time steps before the current time step which will take part in
            backpropogating the gradient through time.
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

        self.W_xh = np.random.uniform(
                                    -np.sqrt(1. / vocab_size), np.sqrt(1. / vocab_size), (self.vocab_size, self.units))
        self.W_hh = np.random.uniform(-np.sqrt(1. / units), np.sqrt(1. / units), (self.units, self.units))
        self.W_hy = np.random.uniform(-np.sqrt(1. / units), np.sqrt(1. / units), (self.units, self.vocab_size))

        self.params = self.W_hh.size + self.W_xh.size + self.W_hy.size

    def forward(self, X, train=False):
        """
        Forward pass the output of the previous layer by using the current layer's weights.

        Parameters
        ----------
        X : numpy.ndarray
            The ouput of the previous layer

        Returns
        -------
        numpy.array : The output of the perceptron.

        Notes
        -----
        Dimensions
        x = batch_size * time_steps(words) * vocab_size
        xt = batch_size * vocab_size
        W_xh = vocab_size * units
        z_t = x * W_xh = batch_size * time_steps * units
        W_hh = units * units
        W_hy = units * vocab_size
        ht = units
        h = batch_size * time_steps(words) * self.units
        """

        self.time_steps = X.shape[1]
        bs = X.shape[0]

        self.input = X

        self.h = np.zeros((bs, self.time_steps + 1, self.units))
        self.z = copy.copy(self.h)

        self.outputs = np.zeros((bs, self.time_steps, self.vocab_size))

        for t in range(self.time_steps):

            # TODO do something about the large input matrix
            self.z[:, t] = np.dot(X[:, t], self.W_xh)
            # for ix in range(X.shape[0]):
            #     self.z[ix, t] += self.W_xh[X[ix, t]]

            self.z[:, t] += np.dot(self.h[:, t - 1], self.W_hh)
            self.h[:, t] = Activations.tanh(self.z[:, t])
            y_t = np.dot(self.h[:, t], self.W_hy)
            self.outputs[:, t] = y_t

        return self.outputs

    def back_propogate(self, dEdO):
        """
        Backpropogate the error, this function adds the share of rnn layer to the accumulated dEdO.

        Parameters
        ----------
        dEdO : numpy.ndarray
            The accumulated dEdO used for calculating error gradient with respect to parameters.

        Returns
        -------
        numpy.array : The accumulated dEdO.

        Notes
        -----
        dEdO -> time_steps, vocab
        h     -> time_steps, units
        """

        self.dEdW_hy = np.zeros_like(self.W_hy)
        self.dEdW_hh = np.zeros_like(self.W_hh)
        self.dEdW_xh = np.zeros_like(self.W_xh)

        for t in reversed(range(self.time_steps)):
            _, dhdz = Activations.tanh(self.z[:, t], return_deriv=True)

            dEdZ = np.dot(dEdO[:, t], self.W_hy.T) * dhdz
            self.dEdW_hy += np.dot(self.h[:, t].T, dEdO[:, t])

            dEdO[:, t] = np.dot(dEdZ, self.W_xh.T)

            memory_indices = np.arange(t - self.memory_window, t + 1)
            memory_indices = memory_indices[memory_indices >= 0]

            for tt in memory_indices:
                _, dhdz = Activations.tanh(self.z[:, tt], return_deriv=True)

                # TODO do something about the large input matrix
                self.dEdW_xh += np.dot(self.input[:, tt].T, np.dot(dhdz, self.W_hh) * dEdZ)
                # for ix in range(self.input.shape[0]):
                #     self.dEdW_xh[self.input[ix, t]] += np.dot(dhdz, self.W_hh)[ix] * dEdZ[ix]

                self.dEdW_hh += np.dot(self.h[:, tt - 1].T, np.dot(dhdz, self.W_hh) * dEdZ)

            # TODO do something about the large input matrix
            self.dEdW_xh += np.dot(self.input[:, t].T, dEdZ)
            # for ix in range(self.input.shape[0]):
            #     self.dEdW_xh[self.input[ix, t]] += dEdZ[ix]

            self.dEdW_hh += np.dot(self.h[:, t - 1].T, dEdZ)

        return dEdO

    def optimize(self, optimizer):
        """
        Updating the layer's weights.

        Parameters
        ----------
        optimizer : fromscratchtoml.neural_network.optimizers
            The optimizer used for updating weights.

        """
        if self.xh_optim is None:
            self.xh_optim = copy.deepcopy(optimizer)
            self.hh_optim = copy.deepcopy(optimizer)
            self.hy_optim = copy.deepcopy(optimizer)

        self.W_xh = self.xh_optim.update_weights(self.W_xh, self.dEdW_xh)
        self.W_hh = self.hh_optim.update_weights(self.W_hh, self.dEdW_hh)
        self.W_hy = self.hy_optim.update_weights(self.W_hy, self.dEdW_hy)
