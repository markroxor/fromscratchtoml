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
    >>> X2 = Distribution.radial_binary(pts=50, mean=[0, 0], start=4, end=5)

    """

    @staticmethod
    def linear(pts=10,
               mean=[0, 2],
               covr=[[0.8, 0.6], [0.6, 0.8]],
               seed=None):
        """Returns a N-D multivariate normal distribution using mean and covariance matrix.

        Parameters
        ----------
        pts : int
              The number of N-D samples to be returned.
        mean : N-D list
               A list of mean values of each dimension.
        covr : NxN D list of list
               The covariance matrix.
        seed : int, optional
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

    @staticmethod
    def radial_binary(pts=100,
               mean=[0, 0],
               start=1,
               end=2,
               seed=None):
        """Returns a radial data distribution (donut shaped) from given mean and
           radial radii.

        Parameters
        ----------
        pts : int
              The number of N-D samples to be returned.
        mean : list
               The centre of the radial distribution.
        start : int
               The lower limit from which radial distribution will start to form.
        end : int
               The upper limit at which radial distribution will end.
        seed : int, optional
               The random state seed value for mantaining reproducability of random
               number generator of numpy.

        Returns
        -------
        _ : pts x 2-D torch.Tensor
            The radial distribution.

        """
        if seed:
            np.random.seed(seed)

        r_theta = np.random.uniform([start, 0], [end, 2 * np.pi], [pts, 2])

        r = r_theta[..., 0]
        theta = r_theta[..., 1]

        dist = np.stack((r * np.cos(theta), r * np.sin(theta)), axis=-1)

        return ch.Tensor(dist)
