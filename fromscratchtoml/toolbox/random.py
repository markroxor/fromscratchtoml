#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import numpy as np
import torch as ch


class Distribution:
    """Objects of this class are the various distributions.

    Examples
    --------
    >>> from fromscratchtoml.toolbox.random import Distribution
    >>> import torch as ch
    >>> X1 = Distribution.linear(pts=50, mean=[8, 20], covr=[[1.5, 1], [1, 2]])

    """

    @staticmethod
    def linear(pts=10,
               mean=[0, 2],
               covr=[[0.8, 0.6], [0.6, 0.8]],
               seed=None):
        """Returns a N-D multivariate normal distribution using mean and covariance matrix.

        Parameters
        ----------
        pts : int.
              The number of N-D samples to be returned.
        mean : N-D list.
               A list of mean values of each dimension.
        covr : NxN D list of list.
               The covariance matrix.
        seed : int, optional.
               The random state seed value for mantaining reproducability of random
               number generator of numpy.

        Returns
        -------
        _ : pts x N-D torch.Tensor
            The multivariate distribution generated from mean and covariance matrix.

        """
        if seed:
            np.random.seed(seed)

        return ch.Tensor(np.random.multivariate_normal(mean, covr, pts))
