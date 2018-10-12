#!/usr/bin/env python
# -*- coding: utf-8 -*-
#



from fromscratchtoml import np

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class l1(object):
    """
    The l1 regularization.
    """

    def __init__(self, lmda=0.01):
        """
        Initialising the lamda value.

        Parameters
        ----------
        lmda : float
            The regularization coefficient.

        """
        self.lmda = lmda

    def value(self, weights, batch_size):
        return self.lmda * np.absolute(weights) / (1.0 * batch_size)

    def grad(self, weights, batch_size):
        return self.lmda / (1.0 * batch_size)


class l2(object):
    """
    The l2 regularization.
    """

    def __init__(self, lmda=0.01):
        """
        Initialising the lamda value.

        Parameters
        ----------
        lmda : float
            The regularization coefficient.

        """
        self.lmda = lmda

    def value(self, weights, batch_size):
        return (self.lmda / 2.0) * np.square(weights) / (1.0 * batch_size)

    def grad(self, weights, batch_size):
        return self.lmda * weights / (1.0 * batch_size)


class l1_l2(object):
    """
    The l1_l2 regularization AKA elastic net.
    """

    def __init__(self, lmda1=0.01, lmda2=0.01):
        """
        Initialising the lamda value.

        Parameters
        ----------
        lmda1 : float
            The l1 regularization coefficient.
        lmda2 : float
            The l2 regularization coefficient.

        """
        self.l1 = lmda1
        self.l2 = lmda2

    def value(self, weights, batch_size):
        return self.l1 * np.absolute(weights) / (1.0 * batch_size) + (self.l2 / 2.0) *\
        np.square(weights) / (1.0 * batch_size)

    def grad(self, weights, batch_size):
        return self.l1 / (1.0 * batch_size) + self.l2 * weights / (1.0 * batch_size)
