#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html
from fromscratchtoml import np


def linear(x, y, **kwargs):
    """The linear kernel.
    https://www.svm-tutorial.com/2014/10/svm-linear-kernel-good-text-classification/
    When to use linear kernel -
    https://stats.stackexchange.com/a/73156/198917

    More on kernel trick -
    https://nlp.stanford.edu/IR-book/html/htmledition/nonlinear-svms-1.html

    Parameters
    ----------
    x : numpy.ndarray
    y : numpy.ndarray

    Returns
    -------
    kernel_trick : float
                   Returns the linear kernel trick which doesn't increase the
                   dimensionality.

    """

    return np.dot(x, y)


def polynomial(x, y, const=0, degree=1, **kwargs):
    """The polynomial kernel. It introduces non-linearity into the margin.
    Visualizing polynomial kernel
    https://www.youtube.com/watch?v=3liCbRZPrZA

    Parameters
    ----------
    x : numpy.ndarray
    y : numpy.ndarray
    const : int, optional
            a polynomial kernel parameter
    degree : int, optional
             the degree of the kernel polynomial.

    Returns
    -------
    kernel_trick : numpy.ndarray
                   Returns the kernel trick to transform the data into `degree`
                   dimensions.

    """

    return pow((np.dot(x, y) + const), degree)


def rbf(x, y, gamma=0.1, **kwargs):
    """The radial basis function or gaussian kernel. Results in closed loop
       margin of higher dimension.
    http://pages.cs.wisc.edu/~matthewb/pages/notes/pdf/svms/RBFKernel.pdf

    Parameters
    ----------
    x : numpy.ndarray
    y : numpy.ndarray
    gamma : float, optional
            a RBF parameter

    Returns
    -------
    kernel_trick : numpy.ndarray
                   Returns the kernel trick to transform the data into infinite
                   dimensions.

    """
    euclidean_dist = pow(np.linalg.norm(x - y), 2)
    return np.exp(- euclidean_dist / (2 * pow(gamma, 2)))
