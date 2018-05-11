#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt  # noqa:F402


def binary_visualize(X, y=None, clf=None, coarse=10, xlabel="x", ylabel="y",
                    title="2D visualization", draw_contour=False, color_seed=1980):
    """Plots the scatter plot of 2D data, along with the margins if clf is provided.

    Parameters
    ----------
    X : numpy.ndarray
        The input 2D data to be plotted.
    y : numpy.ndarray, optional
        The corresponding labels. If not provided, will be predicted.
    clf : a classifier object, optional
        The classifier which forms a basis for plotting margin. Should be passed in case of
        supervised algorithms only.
    coarse: int, optional
        the granula3rity of the separating boundries.
    xlabel: str, optional
        The label of the x axis.
    ylabel: str, optional
        The label of the y axis.
    title: str, optional
        The title of the plot.
    multicolor_contour: bool, optional
        If set to True, each boundry has a different color. Particularly for multi-class SVC.
    color_seed: int, optional
        The seed value for randomising plot colors.

    """
    np.random.seed(color_seed)

    if clf:
        y = clf.predict(X)

    # Also handles the cases when y is None
    unq, y = np.unique(y, return_inverse=True)

    color_intensity = (0.5, 1)
    colors = [(np.random.uniform(*color_intensity), np.random.uniform(*color_intensity),
               np.random.uniform(*color_intensity)) for i in range(len(unq))]

    colored_y = [colors[y[i]] for i in range(len(y))]

    # to auto-get the plot limits this pseudo scatter plot is plotted
    plt.scatter(X[:, 0], X[:, 1], c=colored_y, edgecolors='k')
    xlim = plt.xlim()
    ylim = plt.ylim()

    x_min, x_max = xlim
    y_min, y_max = ylim

    if clf is not None:
        _X, _Y = np.meshgrid(np.arange(x_min, x_max, 1 / (coarse * 1.0)), np.arange(y_min, y_max, 1 / (coarse * 1.0)))

        Z = np.c_[_X.ravel(), _Y.ravel()]
        Z = clf.predict(Z)

        unq, Z = np.unique(Z, return_inverse=True)
        Z = Z.reshape(_X.shape)

        _colors = colors
        if len(_colors) < len(unq) + 1:
            _colors = [(np.random.uniform(*color_intensity), np.random.uniform(*color_intensity),
                       np.random.uniform(*color_intensity)) for i in range(1 + len(unq) - len(colors))] + _colors[::-1]

        cMap = matplotlib.colors.ListedColormap(_colors)

        plt.contourf(_X, _Y, Z, cmap='Pastel1')

    plt.scatter(X[:, 0], X[:, 1], c=colored_y, cmap='Pastel1', edgecolors='k')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.grid()
    plt.show()
