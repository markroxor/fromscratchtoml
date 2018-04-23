#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import torch as ch
import numpy as np

import matplotlib.pyplot as plt


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


def binary_visualize(X1_train, X2_train, clf=None, coarse=10):
    x_min = min(ch.min(X1_train[:, 0]), ch.min(X2_train[:, 0])) * 0.8
    x_max = max(ch.max(X1_train[:, 0]), ch.max(X2_train[:, 0])) * 1.2
    y_min = min(ch.min(X1_train[:, 1]), ch.min(X2_train[:, 1])) * 0.8
    y_max = max(ch.max(X1_train[:, 1]), ch.max(X2_train[:, 1])) * 1.2

    X1_train = X1_train.numpy()
    X2_train = X2_train.numpy()

    plt.plot(X1_train[:, 0], X1_train[:, 1], "mx")
    plt.plot(X2_train[:, 0], X2_train[:, 1], "bo")

    if clf is not None:
        X, Y = np.meshgrid(np.linspace(x_min, x_max, coarse), np.linspace(y_min, y_max, coarse))
        _X = np.array([[x, y] for x, y in zip(np.ravel(X), np.ravel(Y))])

        _, Z = clf.predict(ch.Tensor(_X), return_projection=True)
        Z = Z.view(X.shape)

        plt.contour(X, Y, Z, [0.0], colors='k')

    plt.show()
