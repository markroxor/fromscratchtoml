#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

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
    def __init__(self, C=1000, kernel="linear", **kwargs):
        self.C = C
        self.kernel = partial(getattr(kernels, kernel), **kwargs)

    def __create_kernel_matrix(self, X):
        tricked_x = [self.kernel(X[i], X[j]) for i in range(self.n) for j in range(self.n)]
        tricked_x = ch.Tensor(tricked_x).view(self.n, self.n)
        return tricked_x

    def fit(self, X, y, eta=1e-5):
        X = ch.Tensor(X)
        y = ch.Tensor(y)
        self.n = y.size()[0]

        # create a gram matrix by taking the outer product of y
        gram_matrix_y = ch.ger(y, y)
        K = self.__create_kernel_matrix(X)
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

        lagrange_multipliers = ch.Tensor(list(cvxopt.solvers.qp(P, q, G, h, A, b)['x']))
        effective_lagrange_multiplier_indices = lagrange_multipliers > eta

        self.support_vectors = ch.stack([x for multiplier, x in zip(lagrange_multipliers, X) if multiplier > eta])
        self.support_vectors_y = ch.Tensor(y[effective_lagrange_multiplier_indices])
        self.effective_lagrange_multipliers = lagrange_multipliers[effective_lagrange_multiplier_indices]

        effective_lagrange_multiplier_indices = effective_lagrange_multiplier_indices.nonzero().view(-1)

        self.b = 0
        for i in range(self.support_vectors_y.size()[0]):
            kernel_trick = K[[effective_lagrange_multiplier_indices[i]], effective_lagrange_multiplier_indices]

            self.b += self.support_vectors_y[i] - ch.sum(self.effective_lagrange_multipliers *
                      self.support_vectors_y * kernel_trick)

        self.b /= self.support_vectors_y.size()[0]

    def predict(self, X, return_projection=False):
        if not hasattr(self, "b"):
            raise ModelNotFittedError("Predict called before fitting the model.")

        projections = ch.zeros(X.size()[0])
        for j, x in enumerate(X):
            projection = self.b
            for i in range(self.support_vectors_y.size()[0]):
                projection += self.effective_lagrange_multipliers[i] * self.support_vectors_y[i] *\
                              self.kernel(self.support_vectors[i], x)
            projections[j] = projection

        if return_projection:
            return ch.sign(ch.Tensor(projections)), projections

        return ch.sign(ch.Tensor(projections))
