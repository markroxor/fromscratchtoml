#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import torch as ch
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt # noqa:F402


def sigmoid(x):
    """Returns the sigmoid of x.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    sigmoid of x

    """
    return 1.0 / (1.0 + ch.exp(-x))


def deriv_sigmoid(x):
    """Returns the derivative of sigmoid of x.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    derivative of sigmoid of x

    """
    return sigmoid(x) * (1 - sigmoid(x))


def binary_visualize(X, clf=None, coarse=10):
    """Plots the scatter plot of binary classes, along with the margins if
    clf is provided.

    Parameters
    ----------
    X : an N-D torch.Tensor
        Acts as a generator for each class. These are also plotted.
    clf : a fromscratchtoml.models object, optional
          The classifier which forms a basis for plotting margin.
    coarse: int, optional
            the sections in which the margin is divided in the plot or the
            coarseness of margin.

    """
    x_min, x_max = ch.min(ch.cat(X)[:, 0]), ch.max(ch.cat(X)[:, 0])
    y_min, y_max = ch.min(ch.cat(X)[:, 1]), ch.max(ch.cat(X)[:, 1])

    for x in X:
        plt.scatter(x[:, 0], x[:, 1])

    if clf is not None:
        X, Y = np.meshgrid(np.linspace(x_min, x_max, coarse), np.linspace(y_min, y_max, coarse))
        _X = np.array([[x, y] for x, y in zip(np.ravel(X), np.ravel(Y))])

        _, Z = clf.predict(ch.Tensor(_X), return_projection=True)
        Z = Z.view(X.shape)

        plt.contour(X, Y, Z, [0.0], colors='k')

    plt.show()
