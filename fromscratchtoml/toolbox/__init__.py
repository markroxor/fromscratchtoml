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


def binary_visualize(X, y=None, clf=None, coarse=10, xlim=None, ylim=None, xlabel=None,
                     ylabel=None, title=None, multicolor_contour=False, color_seed=100):
    """Plots the scatter plot of 2D data, along with the margins if clf is provided.

    Parameters
    ----------
    X : an 2xN torch.Tensor
        The input 2D data to be plotted.
    y : an 2xN torch.Tensor, optional
        The corresponding labels. If not provided, will be predicted.
    clf : a fromscratchtoml.models object, optional
          The classifier which forms a basis for plotting margin.
    coarse: int, optional
            the sections in which the margin is divided in the plot or the
            coarseness of margin.

    """
    np.random.seed(color_seed)

    if y is None:
        if clf:
            y = clf.predict(ch.Tensor(X))
            _, y = np.unique(y, return_inverse=True)
        else:
            y = 'b'
    else:
        _, y = np.unique(y, return_inverse=True)

    plt.scatter(X[:, 0], X[:, 1], c=y)

    if xlim is None:
        xlim = plt.xlim()

    if ylim is None:
        ylim = plt.ylim()

    x_min, x_max = xlim
    y_min, y_max = ylim

    if clf is not None:
        clf = clf.classifiers
        colors = [(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)) for i in range(len(clf))]

        for i, _clf in enumerate(clf):
            _X, _Y = np.meshgrid(np.linspace(x_min, x_max, coarse), np.linspace(y_min, y_max, coarse))
            Z = np.array([[_x, _y] for _x, _y in zip(np.ravel(_X), np.ravel(_Y))])
            _, Z = _clf.predict(ch.Tensor(Z), return_projection=True)
            Z = Z.view(_X.shape)

            if multicolor_contour is True:
                plt.contour(_X, _Y, Z, [0.0], colors=[colors[i]])
            else:
                plt.contour(_X, _Y, Z, [0.0], colors='k')

    if xlabel:
        plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)

    if title:
        plt.title(title)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid()
    plt.show()
