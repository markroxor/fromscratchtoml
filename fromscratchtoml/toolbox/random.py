#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import numpy as np
import torch as ch


class Distribution:
    @staticmethod
    def linear(pts1=10, pts2=10,
               mean1=np.array([0, 2]), mean2=np.array([2, 0]),
               cov=np.array([[0.8, 0.6], [0.6, 0.8]])):
        # generate training data in the 2-d case

        X1 = np.random.multivariate_normal(mean1, cov, pts1)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, pts2)
        y2 = np.ones(len(X2)) * -1
        return ch.Tensor(X1), ch.Tensor(y1), ch.Tensor(X2), ch.Tensor(y2)

    @staticmethod
    def non_linear(pts1=10, pts2=10,
                   mean1=[-1, 2], mean2=[1, -1],
                   mean3=[4, -4], mean4=[-4, 4],
                   cov=[[1.0, 0.8], [0.8, 1.0]]):

        X1 = np.random.multivariate_normal(mean1, cov, pts1 / 2)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, pts1 - pts1 / 2)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, pts2 / 2)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, pts2 - pts2 / 2)))
        y2 = np.ones(len(X2)) * -1
        return ch.Tensor(X1), ch.Tensor(y1), ch.Tensor(X2), ch.Tensor(y2)

    @staticmethod
    def linear_overlapping():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 10)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 10)
        y2 = np.ones(len(X2)) * -1
        return ch.Tensor(X1), ch.Tensor(y1), ch.Tensor(X2), ch.Tensor(y2)
