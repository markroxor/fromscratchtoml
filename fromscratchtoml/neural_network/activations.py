#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under The MIT License - https://opensource.org/licenses/MIT

from fromscratchtoml import np

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Activations(object):
    @staticmethod
    def linear(x, alpha=1, return_deriv=False):
        """Returns the scaled input back.

        Parameters
        ----------
        x : numpy.ndarray
            the input
        alpha: int, optional
            the scaling factor
        return_deriv : bool, optional
            if True, returns the derivative of the output along with the output.

        Returns
        -------
        numpy.ndarray : the scaled input.

        """
        if return_deriv:
            return x * alpha, alpha

        return x * alpha

    @staticmethod
    def sigmoid(x, return_deriv=False):
        """Returns the sigmoid of x.

        Parameters
        ----------
        x : numpy.ndarray
            the input
        return_deriv : bool, optional
            if True, returns the derivative of the output along with the output.

        Returns
        -------
        numpy.ndarray : sigmoid of x
        """
        x = np.clip(x, -100, 100)
        _sigmoid = 1. / (1. + np.exp(-x))

        if return_deriv:
            return _sigmoid, _sigmoid * (1 - _sigmoid)

        return _sigmoid

    @staticmethod
    def tanh(x, return_deriv=False):
        """Returns the hyperbolic tan of x.

        Parameters
        ----------
        x : numpy.ndarray
            the input
        return_deriv : bool, optional
            if True, returns the derivative of the output along with the output.

        Returns
        -------
        numpy.ndarray : tanh of x
        """

        '''
        same as (e^x - e^-x) / (e^x + e^-x)
        '''
        _tanh = 2 * Activations.sigmoid(2 * x) - 1

        if return_deriv:
            _, _tanh_deriv = Activations.sigmoid(2 * x, return_deriv=True)
            _tanh_deriv = 4 * _tanh_deriv
            return _tanh, _tanh_deriv

        return _tanh

    @staticmethod
    def relu(x, return_deriv=False):
        """Returns the REctified Linear Unit.

        Parameters
        ----------
        x : numpy.ndarray
            the input
        return_deriv : bool, optional
            if True, returns the derivative of the output along with the output.

        Returns
        -------
        numpy.ndarray : relu of x
        """
        if return_deriv:
            return np.clip(x, 0, None), np.greater_equal(x, np.zeros_like(x)).astype(np.int64)
        return np.clip(x, 0, None)

    @staticmethod
    def leaky_relu(x, alpha=0.3, return_deriv=False):
        """Returns the leaky - REctified Linear Unit.

        Parameters
        ----------
        x : numpy.ndarray
            the input
        return_deriv : bool, optional
            if True, returns the derivative of the output along with the output.

        Returns
        -------
        numpy.ndarray : leaky relu of x
        """
        if return_deriv:
            return np.where(x >= 0, x, x * alpha), np.where(x >= 0, 1, alpha)

        return np.where(x >= 0, x, x * alpha)

    @staticmethod
    def softmax(x, return_deriv=False):
        """Returns the softmax of x.

        Parameters
        ----------
        x : numpy.ndarray
            the input
        return_deriv : bool, optional
            if True, returns the derivative of the output along with the output.

        Returns
        -------
        numpy.ndarray : softmax of x
        """
        # shifting for numerical stability
        # refer https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative
        x -= np.max(x, axis=-1, keepdims=True)
        x = np.clip(x, -100, 100)

        n = np.exp(x)
        d = np.sum(n, axis=-1, keepdims=True)
        _softmax = n / d

        if return_deriv:
            return _softmax, _softmax * (1. - _softmax)

        return _softmax

    @staticmethod
    def tan(x, return_deriv=False):
        """Returns the tan of x.

        Parameters
        ----------
        x : numpy.ndarray
            the input
        return_deriv : bool, optional
            if True, returns the derivative of the output along with the output.

        Returns
        -------
        numpy.ndarray : tan of x
        """

        if return_deriv:
            return np.tan(x), (1. / np.cos(x)) ** 2

        return np.tan(x)
