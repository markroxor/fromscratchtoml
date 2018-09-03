#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under The MIT License - https://opensource.org/licenses/MIT

import sys
import os
import logging

import numpy as np

from .exceptions import ParameterRequiredException

import matplotlib.pyplot as plt

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HiddenPrints:
    # As-it-is from - https://stackoverflow.com/a/45669280/4982729
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def progress(generator):
    n = len(list(generator))

    for i, g in enumerate(generator):
        length = 40

        if i:
            # i = i + 1
            j = length * (i / (n * 1.0))

            bar = ("[%s%s] %d%% " % ('█' * int(j), ' ' * (length - int(j)), (100 / (length * 1.0)) * j))

            sys.stdout.write('\r' + bar)
            sys.stdout.flush()

        yield g
    i = i + 1
    j = length * (i / (n * 1.0))

    bar = ("[%s%s] %d%% " % ('█' * int(j), ' ' * (length - int(j)), (100 / (length * 1.0)) * j))

    sys.stdout.write('\r' + bar)
    sys.stdout.flush()

def binary_visualize(X, y=None, clf=None, coarse=50, xlabel="x", ylabel="y",
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
    try:
        import cupy
        if isinstance(y, cupy.core.core.ndarray):
            y = cupy.asnumpy(y)
        if isinstance(X, cupy.core.core.ndarray):
            X = cupy.asnumpy(X)
    except ImportError:
        pass

    np.random.seed(color_seed)

    if len(X.shape) != 2 or X.shape[1] != 2:
        logger.warn("Cannot plot {} dimensional data.".format(X.shape[1]))
        return None

    if clf:
        y = clf.predict(X)
        y = cupy.asnumpy(y)

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

    if draw_contour:
        if clf is None:
            raise ParameterRequiredException("Missing required parameter clf for drawing contour.")

        _X, _Y = np.meshgrid(np.arange(x_min, x_max, 1 / (coarse * 1.0)), np.arange(y_min, y_max, 1 / (coarse * 1.0)))

        Z = np.c_[_X.ravel(), _Y.ravel()]
        Z = cupy.asnumpy(clf.predict(Z))

        unq, Z = np.unique(Z, return_inverse=True)
        Z = Z.reshape(_X.shape)

        plt.contourf(_X, _Y, Z, cmap='Pastel1')

    plt.scatter(X[:, 0], X[:, 1], c=colored_y, cmap='Pastel1', edgecolors='k')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.grid()
    plt.show()
