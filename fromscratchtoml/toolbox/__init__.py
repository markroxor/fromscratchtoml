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


def binary_visualize(X, y=None, clf=None, coarse=10, xlabel=None, ylabel=None, title=None,
                     multicolor_contour=False, color_seed=10):
    """Plots the scatter plot of 2D data, along with the margins if clf is provided.

    Parameters
    ----------
    X : an 2xN torch.Tensor
        The input 2D data to be plotted.
    y : an 2xN torch.Tensor, optional
        The corresponding labels. If not provided, will be predicted.
    clf : a classifier object, optional
        The classifier which forms a basis for plotting margin. Should be passed in case of
        supervised algorithms only.
    coarse: int, optional
        the granualirity of the separating boundries.
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
    colors = [(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)) for i in range(len(unq))]

    colored_y = [colors[y[i]] for i in range(len(y))]

    # to auto-get the plot limits this pseudo scatter plot is plotted
    plt.scatter(X[:, 0], X[:, 1], c=colored_y, edgecolors='k')
    xlim = plt.xlim()
    ylim = plt.ylim()

    x_min, x_max = xlim
    y_min, y_max = ylim

    if clf is not None:
        if hasattr(clf, 'classifiers'):
            clf = clf.classifiers
        else:
            clf = [clf]

        for i, _clf in enumerate(clf):

            if type(_clf).__name__ == 'KNeighborsClassifier':
                coarse = 1 / (coarse * 1.0)
                _X, _Y = np.meshgrid(np.arange(x_min, x_max, coarse), np.arange(y_min, y_max, coarse))
                Z = np.c_[_X.ravel(), _Y.ravel()]
                Z = _clf.predict(Z)

                Z = Z.reshape(_X.shape)

                plt.set_cmap(plt.cm.Paired)
                plt.pcolormesh(_X, _Y, Z)

            elif type(_clf).__name__ == 'SVC':
                _X, _Y = np.meshgrid(np.linspace(x_min, x_max, coarse), np.linspace(y_min, y_max, coarse))
                Z = np.array([[_x, _y] for _x, _y in zip(np.ravel(_X), np.ravel(_Y))])
                _, Z = _clf.predict(ch.Tensor(Z), return_projection=True)
                Z = Z.numpy()

                Z = Z.reshape(_X.shape)

                if multicolor_contour is True:
                    plt.contour(_X, _Y, Z, [0.0], colors=[colors[i]])
                else:
                    plt.contour(_X, _Y, Z, [0.0], colors='k')

    plt.scatter(X[:, 0], X[:, 1], c=colored_y, edgecolors='k')

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
