#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0

import cvxopt
import torch as ch
import numpy as np

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
    >>> _, projections = clf_lin.predict(X, return_projection=True)

    """

    def __init__(self, C=1000, kernel="linear", **kwargs):
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
        self.kernel = partial(getattr(kernels, kernel), **kwargs)

    def create_kernel_matrix(self, X):
        """Creates a gram kernel matrix of training data.
        Refer - https://en.wikipedia.org/wiki/Gramian_matrix

        Parameters
        ----------
        X : torch.Tensor

        Returns
        -------
        tricked_x : torch.Tensor
                    The gram kernel matrix.

        """

        tricked_x = [self.kernel(X[i], X[j]) for i in range(self.n)
                     for j in range(self.n)]

        tricked_x = ch.Tensor(tricked_x).view(self.n, self.n)
        return tricked_x

    def fit(self, X, y, eta=1e-5):
        """Fits the svc model on training data.

        Parameters
        ----------
        X : torch.Tensor
            The training features.
        y : torch.Tensor
            The training labels.
        eta : float
              the threshold for selecting lagrange multipliers.

        """

        X = ch.Tensor(X)
        y = ch.Tensor(y)
        self.n = y.size()[0]

        # create a gram matrix by taking the outer product of y
        gram_matrix_y = ch.ger(y, y)
        K = self.create_kernel_matrix(X)
        gram_matrix_xy = gram_matrix_y * K

        P = cvxopt.matrix(gram_matrix_xy.numpy().astype(np.double))
        q = cvxopt.matrix(-ch.ones(self.n).numpy().astype(np.double))

        G1 = cvxopt.spmatrix(-1.0, range(self.n), range(self.n))
        G2 = cvxopt.spmatrix(1, range(self.n), range(self.n))
        G = cvxopt.matrix([[G1, G2]])

        h1 = cvxopt.matrix(ch.zeros(self.n).numpy().astype(np.double))
        h2 = cvxopt.matrix(ch.ones(self.n).numpy().astype(np.double) * self.C)
        h = cvxopt.matrix([[h1, h2]])

        A = cvxopt.matrix(y.numpy().astype(np.double)).trans()
        b = cvxopt.matrix(0.0)

        lagrange_multipliers = ch.Tensor(list(cvxopt.solvers.qp(P, q, G, h, A,
                                                                b)['x']))

        lagrange_multiplier_indices = lagrange_multipliers > eta

        self.support_vectors = ch.stack([x for multiplier, x in
                                         zip(lagrange_multipliers, X)
                                         if multiplier > eta])

        self.support_vectors_y = ch.Tensor(y[lagrange_multiplier_indices])
        self.lagrange_multipliers = lagrange_multipliers[lagrange_multiplier_indices]

        lagrange_multiplier_indices = lagrange_multiplier_indices.nonzero().view(-1)

        self.b = 0
        for i in range(self.support_vectors_y.size()[0]):
            kernel_trick = K[[lagrange_multiplier_indices[i]], lagrange_multiplier_indices]

            self.b += self.support_vectors_y[i] - ch.sum(self.lagrange_multipliers *
                      self.support_vectors_y * kernel_trick)

        self.b /= self.support_vectors_y.size()[0]

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
            for i in range(self.support_vectors_y.size()[0]):
                projection += self.lagrange_multipliers[i] * self.support_vectors_y[i] *\
                              self.kernel(self.support_vectors[i], x)
            projections[j] = projection

        if return_projection:
            return ch.sign(ch.Tensor(projections)), projections

        return ch.sign(ch.Tensor(projections))
