#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import cvxopt
import torch as ch

from functools import partial

from fromscratchtoml.models.base_model import BaseModel
from fromscratchtoml.toolbox import kernels
from fromscratchtoml.toolbox.exceptions import ModelNotFittedError

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
    >>> Y1 = ch.ones(X1.size()[0])
    >>> Y2 = -ch.ones(X2.size()[0])
    >>> X = ch.cat((X1, X2))
    >>> y = ch.cat((Y1, Y2))
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

    def __create_kernel_matrix(self, X):
        """Creates a gram kernel matrix of training data.
        Refer - https://en.wikipedia.org/wiki/Gramian_matrix

        Parameters
        ----------
        X : torch.Tensor

        Returns
        -------
        kernel_matrix : torch.Tensor
                    The gram kernel matrix.

        """

        kernel_matrix = [self.kernel(X[i], X[j]) for i in range(self.n)
                     for j in range(self.n)]

        kernel_matrix = ch.Tensor(kernel_matrix).view(self.n, self.n)
        return kernel_matrix

    def fit(self, X, y, multiplier_threshold=1e-5):
        """Fits the svc model on training data.

        Parameters
        ----------
        X : torch.Tensor
            The training features.
        y : torch.Tensor
            The training labels.
        multiplier_threshold : float
              the threshold for selecting lagrange multipliers.

        """

        X = ch.Tensor(X)
        y = ch.Tensor(y)
        self.n = y.size()[0]

        # create a gram matrix by taking the outer product of y
        gram_matrix_y = ch.ger(y, y)
        K = self.__create_kernel_matrix(X)
        gram_matrix_xy = gram_matrix_y * K

        P = cvxopt.matrix(gram_matrix_xy.numpy().astype(float))
        q = cvxopt.matrix(-ch.ones(self.n).numpy().astype(float))

        G1 = cvxopt.spmatrix(-1.0, range(self.n), range(self.n))
        G2 = cvxopt.spmatrix(1, range(self.n), range(self.n))
        G = cvxopt.matrix([[G1, G2]])

        h1 = cvxopt.matrix(ch.zeros(self.n).numpy().astype(float))
        h2 = cvxopt.matrix(ch.ones(self.n).numpy().astype(float) * self.C)
        h = cvxopt.matrix([[h1, h2]])

        A = cvxopt.matrix(y.numpy().astype(float)).trans()
        b = cvxopt.matrix(0.0)

        lagrange_multipliers = ch.Tensor(list(cvxopt.solvers.qp(P, q, G, h, A,
                                                                b)['x']))

        lagrange_multiplier_indices = lagrange_multipliers.ge(multiplier_threshold)
        lagrange_multiplier_indices = lagrange_multiplier_indices.nonzero().view(-1)

        self.support_vectors = X.index_select(0, lagrange_multiplier_indices)
        self.support_vectors_y = y.index_select(0, lagrange_multiplier_indices)
        self.support_lagrange_multipliers = lagrange_multipliers.index_select(0, lagrange_multiplier_indices)

        self.b = 0
        self.n_support_vectors = self.support_vectors.size()[0]

        for i in range(self.n_support_vectors):
            kernel_trick = K[[lagrange_multiplier_indices[i]], lagrange_multiplier_indices]

            self.b += self.support_vectors_y[i] - ch.sum(self.support_lagrange_multipliers *
                      self.support_vectors_y * kernel_trick)

        self.b /= self.n_support_vectors

    def predict(self, X, return_projection=False):
        """Predicts the class of input test data.

        Parameters
        ----------
        x : torch.Tensor
            The test data which is to be classified.
        return_projection : bool, optional
                            returns the projection of test data on the margin
                            along with the class prediction.

        Returns
        -------
        prediction : torch.Tensor
                     A torch.Tensor consisting of 1, 0, -1 denoting the class of
                     test data
        projections : torch.Tensor
                      The projection formed by the test data point on the
                      margin.

        """

        if not hasattr(self, "b"):
            raise ModelNotFittedError("Predict called before fitting the model.")

        projections = ch.zeros(X.size()[0])

        for j, x in enumerate(X):
            projection = self.b
            for i in range(self.n_support_vectors):
                projection += self.support_lagrange_multipliers[i] * self.support_vectors_y[i] *\
                              self.kernel(self.support_vectors[i], x)
            projections[j] = projection

        if return_projection:
            return ch.sign(ch.Tensor(projections)), projections

        return ch.sign(ch.Tensor(projections))
