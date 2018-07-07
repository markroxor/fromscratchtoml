#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html
import sys
import logging

import numpy as np

from .exceptions import ParameterRequiredException

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt  # noqa:F402

#bokeh
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.palettes import Set3


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def binary_visualize_bokeh(X, y=None, clf=None, coarse=50, xlabel="x", ylabel="y",
title="2D visualization", draw_contour=False, color_seed=1980, plot=True):
    """Plots the interactive scatter plot of 2D data, along with the margins if clf is provided with bokeh.
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

    if len(X.shape) != 2 or X.shape[1] != 2:
        logger.warn("Cannot plot {} dimensional data.".format(X.shape[1]))
        return None

    if clf:
        y = clf.predict(X)

    # Also handles the cases when y is None
    unq, y = np.unique(y, return_inverse=True)

    color_intensity = (0.5, 1)
    colors = [(np.random.uniform(*color_intensity), np.random.uniform(*color_intensity),
               np.random.uniform(*color_intensity)) for i in range(len(unq))]

    colored_y = [Set3[12][y[i]] for i in range(len(y))]
    line_color = ['black' for i in range(len(y))]



    source = ColumnDataSource(data=dict(x=X[:, 0], y=X[:, 1], color=colored_y, line_color=line_color))

    TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
    ]

    p = figure(plot_height=400, plot_width=600, tooltips=TOOLTIPS, title=title)
    p.circle('x','y', radius=0.1, fill_color='color', source=source, line_color='line_color')

    if(plot):
        show(p)








def binary_visualize(X, y=None, clf=None, coarse=50, xlabel="x", ylabel="y",
title="2D visualization", draw_contour=False, color_seed=1980, plot=True):
    """Plots the interactive scatter plot of 2D data, along with the margins if clf is provided with bokeh.
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

    if len(X.shape) != 2 or X.shape[1] != 2:
        logger.warn("Cannot plot {} dimensional data.".format(X.shape[1]))
        return None

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

    if draw_contour:
        if clf is None:
            raise ParameterRequiredException("Missing required parameter clf for drawing contour.")

        _X, _Y = np.meshgrid(np.arange(x_min, x_max, 1 / (coarse * 1.0)), np.arange(y_min, y_max, 1 / (coarse * 1.0)))

        Z = np.c_[_X.ravel(), _Y.ravel()]
        Z = clf.predict(Z)

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

    if plot:
        plt.show()
