#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import cvxopt
import torch as ch
import numpy as np

from fromscratchtoml.models.base_model import BaseModel
from fromscratchtoml.toolbox.kernels import Kernel
from fromscratchtoml.toolbox.exceptions import ModelNotFittedError

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

cvxopt.solvers.options['show_progress'] = False


class SVC(BaseModel):
    def __init__(self, kernel="linear", **kwargs):
        self.kernel = getattr(Kernel(**kwargs), kernel)()

    def __apply_kernel_trick(self, X):
        tricked_x = [self.kernel(X[i], X[j]) for i in range(self.n) for j in range(self.n)]
        tricked_x = ch.Tensor(tricked_x).view(self.n, self.n)
        return tricked_x

    def __sign(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        return x

    def fit(self, X, y, eta=1e-5):
        X = ch.Tensor(X)
        y = ch.Tensor(y)
        self.n = y.size()[0]

        # create a gram matrix by taking the outer product of y
        gram_matrix_y = ch.ger(y, y)
        K = ch.zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                K[i, j] = self.kernel(X[i], X[j])

        # a = self.__apply_kernel_trick(X)
        gram_matrix_xy = gram_matrix_y * K

        P = cvxopt.matrix(gram_matrix_xy.numpy().astype(np.double))
        q = cvxopt.matrix(-ch.ones(self.n).numpy().astype(np.double))
        G = cvxopt.spmatrix(-1.0, range(self.n), range(self.n))
        h = cvxopt.matrix(ch.zeros(self.n).numpy().astype(np.double))
        A = cvxopt.matrix(y.numpy().astype(np.double)).trans()
        b = cvxopt.matrix(0.0)

        lagrange_multipliers = ch.Tensor(list(cvxopt.solvers.qp(P, q, G, h, A, b)['x']))
        effective_lagrange_multiplier_indices = lagrange_multipliers > eta
        ind = []
        for i in range(effective_lagrange_multiplier_indices.shape[0]):
            # print(effective_lagrange_multiplier_indices)
            if effective_lagrange_multiplier_indices[i] == 1:
                ind.append(i)
        self.support_vectors = ch.stack([x for multiplier, x in zip(lagrange_multipliers, X) if multiplier > eta])
        self.support_vectors_y = ch.Tensor(y[effective_lagrange_multiplier_indices])
        self.effective_lagrange_multipliers = lagrange_multipliers[effective_lagrange_multiplier_indices]

        self.b = 0
        for i in range(self.support_vectors_y.size()[0]):
            a = []
            for j in range(len(ind)):
                a.append(self.kernel(X[ind[i]], X[ind[j]]))

            self.b += self.support_vectors_y[i] - ch.sum(self.effective_lagrange_multipliers *
                      self.support_vectors_y * ch.Tensor(a))

        self.b /= self.support_vectors_y.size()[0]

    def predict(self, x, return_projection=False):
        if not hasattr(self, "b"):
            raise ModelNotFittedError("Predict called before fitting the model.")

        self.prediction = self.b
        for i in range(self.support_vectors_y.size()[0]):
            self.prediction += self.effective_lagrange_multipliers[i] * self.support_vectors_y[i] *\
                         self.kernel(self.support_vectors[i], x)

        logger.info(self.__sign(self.prediction))
        if return_projection:
            return self.__sign(self.prediction), self.prediction

        return self.__sign(self.prediction)
