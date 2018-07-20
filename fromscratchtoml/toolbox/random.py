#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

from fromscratchtoml import np
import numpy

class Distribution:
    """Objects of this class are the various distributions.

    Examples
    --------
    >>> from fromscratchtoml.toolbox.random import Distribution
    >>> X1 = Distribution.linear(pts=50, mean=[8, 20], covr=[[1.5, 1], [1, 2]])
    >>> X2 = Distribution.radial_binary(pts=50, mean=[0, 0], st=4, ed=5)

    """

    @staticmethod
    def linear(pts=10,
               mean=[0, 0],
               covr=[[1, 0], [0, 1]],
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
        _ : pts x N-D np.ndarray
            The multivariate distribution generated from mean and covariance matrix.

        """
        if seed:
            np.random.seed(seed)

        return np.random.multivariate_normal(mean, covr, pts)

    @staticmethod
    def radial_binary(pts=100, mean=[0, 0], st=1, ed=2,
                      tmin=0, tmax=2 * np.pi, seed=None):
        """Returns a radial data distribution (donut shaped) from given mean and
           radial radii.

        Parameters
        ----------
        pts : int, optional
            The number of N-D samples to be returned.

        mean : list, optional
            The centre of the radial distribution.

        st : int, optional
            The lower limit from which radial distribution will start to form.
        ed : int, optional
            The upper limit at which radial distribution will end.

        tmin : float, optional
            The lower limit of theta from which radial distribution starts.
        tmax : float, optional
            The upper limit of theta at which radial distribution ends.

        seed : int, optional
               The random state seed value for mantaining reproducability of random
               number generator of numpy.

        Returns
        -------
        _ : pts x 2-D numpy.ndarray
            The radial distribution.

        """
        if seed:
            numpy.random.seed(seed)

        r_theta = numpy.random.uniform([st, tmin], [ed, tmax], [pts, 2])

        r = r_theta[..., 0]
        theta = r_theta[..., 1]

        dist = numpy.stack((r * numpy.cos(theta) + mean[0], r * numpy.sin(theta) + mean[1]), axis=-1)

        return np.asarray(dist)
