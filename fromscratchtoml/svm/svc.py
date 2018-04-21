#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import cvxopt
import torch as ch
import numpy as np

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

cvxopt.solvers.options['show_progress'] = False


class SVC(object):
    def __init__(self):
        pass

    def __apply_kernel_trick(self, X):
        tricked_x = [ch.dot(X[i], X[j]) for i in range(self.n) for j in range(self.n)]
        tricked_x = ch.Tensor(tricked_x).view(self.n, self.n)
        return tricked_x

    def __sign(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        return x

    def fit(self, X, y, eta=1e-7):
        X = ch.Tensor(X)
        y = ch.Tensor(y)
        self.X = X
        self.n = y.size()[0]

        # create a gram matrix by taking the outer product of y
        gram_matrix_y = ch.ger(y, y)
        gram_matrix_xy = gram_matrix_y * self.__apply_kernel_trick(X)

        P = cvxopt.matrix(gram_matrix_xy.numpy().astype(np.double))
        q = cvxopt.matrix(-ch.ones_like(y).numpy().astype(np.double))
        G = cvxopt.spmatrix(-1.0, range(self.n), range(self.n))
        h = cvxopt.matrix(ch.zeros_like(y).numpy().astype(np.double))
        A = cvxopt.matrix(y.numpy().astype(np.double)).trans()
        b = cvxopt.matrix(0.0)

        lagrange_multipliers = ch.Tensor(list(cvxopt.solvers.qp(P, q, G, h, A, b)['x']))
        effective_lagrange_multiplier_indices = lagrange_multipliers > eta

        self.support_vectors = ch.stack([x for multiplier, x in zip(lagrange_multipliers, X) if multiplier > eta])
        self.support_vectors_y = ch.Tensor(y[effective_lagrange_multiplier_indices])
        self.effective_lagrange_multipliers = lagrange_multipliers[effective_lagrange_multiplier_indices]

        self.w = 0
        for i in range(self.support_vectors_y.size()[0]):
            self.w += self.effective_lagrange_multipliers[i] * self.support_vectors[i] * self.support_vectors_y[i]

        self.b = 0
        for i in range(self.support_vectors_y.size()[0]):
            self.b += self.support_vectors_y[i] - ch.dot(self.w, self.support_vectors[i])

        self.b /= self.support_vectors_y.size()[0]

    def predict(self, X=None):
        for x in self.X:
            logger.info(self.__sign(ch.dot(self.w, x) + self.b))

    def save_model(self, file_path):
        """This function saves the model in a file for loading it in future.

        Parameters
        ----------
        file_path : str
            The path to file where the model should be saved.

        """
        ch.save(self.__dict__, file_path)
        return

    def load_model(self, file_path):
        """This function loads the saved model from a file.

        Parameters
        ----------
        file_path : str
            The path of file from where the model should be retrieved.

        """
        self.__dict__ = ch.load(file_path)
        return
