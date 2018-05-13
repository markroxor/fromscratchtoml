#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import numpy as np

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Activations(object):
    @staticmethod
    def linear(x, alpha=1):
        """Returns the scaled input back.

        Parameters
        ----------
        x : numpy.ndarray
        alpha: int, optional
            the scaling factor

        Returns
        -------
        _ : numpy.ndarray
            x

        """
        return x * alpha

    @staticmethod
    def sigmoid(x, return_deriv=False):
        """Returns the sigmoid of x.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        _ : numpy.ndarray
            sigmoid of x

        """
        _sigmoid = 1.0 / (1.0 + np.exp(-x))

        if return_deriv:
            return _sigmoid, _sigmoid * (1 - _sigmoid)

        return _sigmoid

    @staticmethod
    def tanh(x):
        """Returns the sigmoid of x.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        _ : numpy.ndarray
            tanh of x

        """

        '''
        same as (e^x - e^-x) / (e^x + e^-x)
        '''
        return 2 * Activations.sigmoid(2 * x) - 1

    @staticmethod
    def relu(x):
        """Returns the REctified Linear Unit.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        _ : numpy.ndarray
            relu of x

        """

        return np.clip(x, 0, max(x))

    @staticmethod
    def leaky_relu(x, alpha=0.3):
        """Returns the leaky - REctified Linear Unit.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        _ : numpy.ndarray
            leaky - relu of x

        """
        def leakit(x, alpha):
            if x >= 0:
                return x
            return x * alpha

        # I <3 python
        # perform elementwise ops using nps vectorize
        return np.vectorize(leakit)(x, alpha)

    @staticmethod
    def step(x):
        """Returns the signum of x.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        _ : numpy.ndarray
            signum of x

        """
        def stepit(x):
            if x > 0:
                return 1
            elif x < 0:
                return -1
            return 0

        # perform elementwise ops using nps vectorize
        return np.vectorize(stepit)(x)

    @staticmethod
    def softmax(x):
        """Returns the softmax of x.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        _ : numpy.ndarray
            softmax of x

        """
        n = np.exp(x)
        d = np.sum(np.exp(x))

        return n / d
