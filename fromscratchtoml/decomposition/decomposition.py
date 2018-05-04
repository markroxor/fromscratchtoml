#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import numpy as np

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Decomposition(object):

    @staticmethod
    def cov_matrix(X, Y):
        """Computes the covariance matrix.

        Parameters
        ----------
        X : NxD numpy.ndarray
            the input matrix
        Y : NxD numpy.ndarray
            the input matrix

        Returns
        -------
        cov_matrix : numpy.ndarray
            The covariance matrix

        See Also
        --------
        eigens : Computes eigen values and eigen vectors.

        References
        ----------
        https://www.theanalysisfactor.com/covariance-matrices/

        Examples
        --------
        >>> from fromscratchtoml.decomposition import Decomposition
        >>> X = np.array([[12, 1, 32],[43, 54, 61]])
        >>> Decomposition.cov_matrix(X, X)
        array([[ 480.5,  821.5,  449.5],
        [ 821.5, 1404.5,  768.5],
        [ 449.5,  768.5,  420.5]])
        """

        x_norm = (X - X.mean(0))
        y_norm = (Y - Y.mean(0))
        cov_matrix = np.dot(x_norm.T, y_norm) / (x_norm.shape[0] - 1)
        return cov_matrix

    @staticmethod
    def eigens(X):
        """Returns the eigen values and eigen vectors.

        Parameters
        ----------
        X : NxD numpy.ndarray
            the input matrix

        Returns
        -------
        eigen_vals : numpy.ndarray
            Eigen vectors
        eigen_vecs : numpy.ndarray
            Eigen values

        See Also
        --------
        cov_matrix : Computes the covariance matrix.

        Examples
        --------
        >>> from fromscratchtoml.decomposition import Decomposition
        >>> X = np.array([[12, 1, 32],[43, 54, 61]])
        >>> eigen_vals, eigen_vecs = Decomposition.eigens(X)
        >>> eigen_vals
        array([-2.27373675e-13,  2.30550000e+03,  0.00000000e+00])
        >>> eigen_vecs
        array([[-0.88971082,  0.45652455, -0.19317543],
               [ 0.40049175,  0.78050971, -0.38160513],
               [ 0.21913699,  0.42707135,  0.90391414]])
        """

        cov_matrix = Decomposition.cov_matrix(X, X)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix)

        return eigen_vals, eigen_vecs

    @staticmethod
    def pca(X, num_components=None, return_scaled=False):
        """Returns the eigen values and eigen vectors.

        Parameters
        ----------
        X : NxD numpy.ndarray
            the input matrix
        num_components: int
            number of components of PCA.
        return_scaled: bool, optional
            if set to true, also returns the scaled X in different vector space.

        Returns
        -------
        principal_components : numpy.ndarray
            The principal components of matrix X.
        rescaled_x : numpy.ndarray
            The scaled matrix in different vector space.

        See Also
        --------
        cov_matrix : Computes the covariance matrix.

        Examples
        --------
        >>> from fromscratchtoml.decomposition import Decomposition
        >>> X = np.array([[12, 1, 32],[43, 54, 61]])
        >>> principal_components, rescaled_x = Decomposition.pca(X, return_scaled=True)
        >>> principal_components
        array([[ 0.45652455, -0.19317543, -0.88971082],
               [ 0.78050971, -0.38160513,  0.40049175],
               [ 0.42707135,  0.90391414,  0.21913699]])
        >>> rescaled_x
        array([[-3.39521722e+01,  2.33146835e-15,  3.26128013e-15],
               [ 3.39521722e+01, -2.33146835e-15, -3.26128013e-15]])
        """

        if num_components is None:
            num_components = X.shape[1]
        eigen_vals, eigen_vecs = Decomposition.eigens(X)

        sort_index = np.argsort(-eigen_vals)
        eigen_vals = eigen_vals[sort_index][:num_components]
        principal_components = eigen_vecs[:, sort_index][:, :num_components]

        rescaled_x = np.dot(X - X.mean(0), principal_components)

        if return_scaled:
            return principal_components, rescaled_x

        return principal_components
