#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import cvxopt
# import torch as ch
import numpy as np

from functools import partial

from ..base import BaseModel
from ..toolbox import kernels
from ..toolbox.exceptions import ModelNotFittedError

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

cvxopt.solvers.options['show_progress'] = False


class SVC(BaseModel):
    """This implements support vector machine classification.

    Examples
    --------
    >>> X1 = Distribution.linear(pts=100, mean=[8, 10], covr=[[1.5, 1], [1, 1.5]])
    >>> X2 = Distribution.linear(pts=100, mean=[9, 5], covr=[[1.5, 1], [1, 1.5]])
    >>> Y1 = np.ones(X1.shape[0])
    >>> Y2 = -np.ones(X2.shape[0])
    >>> X = np.cat((X1, X2))
    >>> y = np.cat((Y1, Y2))
    >>> clf_lin = svm.SVC(kernel='linear')
    >>> clf_lin.fit(X, y)
    >>> predictions = clf_lin.predict(X)

    """

    def __init__(self, C=1000, kernel="rbf", gamma=2, **kwargs):
        """Initializing the svc class parameters.

        Parameters
        C : int
            Soft margin's positive box constraint.
        kernel : string
            Kernels like `linear`, `polynomial`, `rbf`
            Refer `fromscratchtoml.toolbox.kernels`
        **kwargs : dictionary
            These parameters will initilize the kernels.
            Refer `fromscratchtoml.toolbox.kernels`

        """
        self.C = C
        self.kernel = partial(getattr(kernels, kernel), gamma=gamma, **kwargs)
        self.classifiers = []

    def __create_kernel_matrix(self, X):
        """Creates a gram kernel matrix of training data.
        Refer - https://en.wikipedia.org/wiki/Gramian_matrix

        Parameters
        ----------
        X : tornp.array

        Returns
        -------
        kernel_matrix : tornp.array
            The gram kernel matrix.

        """

        kernel_matrix = [self.kernel(X[i], X[j]) for i in range(self.n)
                     for j in range(self.n)]

        kernel_matrix = np.array(kernel_matrix).reshape(self.n, self.n)
        return kernel_matrix

    def fit(self, X, y, multiplier_threshold=1e-5):
        """Fits the svc model on training data.

        Parameters
        ----------
        X : tornp.array
            The training features.
        y : tornp.array
            The training labels.
        multiplier_threshold : float
            The threshold for selecting lagrange multipliers.

        Returns
        -------
        kernel_matrix : list of svm.SVC
            A list of all the classifiers used for multi class classification
        """
        X = np.array(X)
        self.y = y
        self.n = self.y.shape[0]

        self.uniques, self.ind = np.unique(self.y, return_index=True)
        self.n_classes = len(self.uniques)

        # Do multi class classification
        if sorted(self.uniques) != [-1, 1]:
            y_list = [np.where(self.y == u, 1, -1) for u in self.uniques]

            for y_i in y_list:
                # Copy the current initializer
                clf = SVC()
                clf.kernel = self.kernel
                clf.C = self.C

                self.classifiers.append(clf.fit(X, y_i))
            return

        # create a gram matrix by taking the outer product of y
        gram_matrix_y = np.outer(self.y, self.y)
        K = self.__create_kernel_matrix(X)
        gram_matrix_xy = gram_matrix_y * K

        P = cvxopt.matrix(gram_matrix_xy)
        q = cvxopt.matrix(-np.ones(self.n))

        G1 = cvxopt.spmatrix(-1.0, range(self.n), range(self.n))
        G2 = cvxopt.spmatrix(1, range(self.n), range(self.n))
        G = cvxopt.matrix([[G1, G2]])

        h1 = cvxopt.matrix(np.zeros(self.n))
        h2 = cvxopt.matrix(np.ones(self.n) * self.C)
        h = cvxopt.matrix([[h1, h2]])

        A = cvxopt.matrix(self.y.astype(np.double)).trans()
        b = cvxopt.matrix(0.0)

        lagrange_multipliers = np.array(list(cvxopt.solvers.qp(P, q, G, h, A,
                                                                b)['x']))

        lagrange_multiplier_indices = np.greater_equal(lagrange_multipliers, multiplier_threshold)
        lagrange_multiplier_indices = list(map(list, lagrange_multiplier_indices.nonzero()))[0]

        # self.support_vectors = np.take(X, lagrange_multiplier_indices, axis=1)
        self.support_vectors = X[lagrange_multiplier_indices]
        # print(X)
        # print(lagrange_multiplier_indices)
        # print(self.support_vectors)
        # self.support_vectors_y = np.take(self.y, lagrange_multiplier_indices)
        self.support_vectors_y = self.y[lagrange_multiplier_indices]
        # self.support_lagrange_multipliers = np.take(lagrange_multipliers, lagrange_multiplier_indices)
        self.support_lagrange_multipliers = lagrange_multipliers[lagrange_multiplier_indices]
        self.b = 0
        self.n_support_vectors = self.support_vectors.shape[0]

        for i in range(self.n_support_vectors):
            kernel_trick = K[[lagrange_multiplier_indices[i]], lagrange_multiplier_indices]

            self.b += self.support_vectors_y[i] - np.sum(self.support_lagrange_multipliers *
                      self.support_vectors_y * kernel_trick)

        self.b /= self.n_support_vectors

        self.classifiers = [self]
        return self

    def predict(self, X, return_projection=False):
        """Predicts the class of input test data.

        Parameters
        ----------
        x : tornp.array
            The test data which is to be classified.
        return_projection : bool, optional
            returns the projection of test data on the margin
            along with the class prediction.

        Returns
        -------
        prediction : tornp.array
            A tornp.array consisting of 1, 0, -1 denoting the class of
            test data
        projections : tornp.array, optional
            The projection formed by the test data point on the
            margin.

        """
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        if len(self.classifiers) == 0:
            raise ModelNotFittedError("Predict called before fitting the model.")

        projections = np.zeros(X.shape[0])
        predictions = np.zeros(X.shape[0], dtype=np.int64)

        # If the input labels are not of as desired by svc i.e - [-1, 1]
        if sorted(self.uniques) != [-1, 1]:
            for j, x in enumerate(X):
                for i, clas in enumerate(self.classifiers):
                    prediction, projection = self.classifiers[i].predict(x, return_projection=True)

                    if int(prediction) == 1:
                        if projections[j] != 0:
                            projections[j] = 0
                            predictions[j] = 0
                            break
                        else:
                            predictions[j] = self.y[self.ind[i]]
                            projections[j] = float(projection)

        else:
            for j, x in enumerate(X):
                projection = self.b
                for i in range(self.n_support_vectors):
                    projection += self.support_lagrange_multipliers[i] * self.support_vectors_y[i] *\
                                  self.kernel(self.support_vectors[i], x)
                projections[j] = projection
                predictions[j] = np.sign(projection)

        if return_projection:
            return predictions, projections

        return predictions
