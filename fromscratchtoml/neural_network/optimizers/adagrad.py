#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under The MIT License - https://opensource.org/licenses/MIT

from fromscratchtoml import np

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# optimizes/updates the weights
class Adagrad(object):
    """
    The adagrad optimizer.
    """

    def __init__(self, learning_rate=0.01):
        """
        Initialising the optimizer parameters.

        Parameters
        ----------
        learning_rate : float
            the rate of change of weights. The higher the learning rate - more is the change in the parameters.
        """
        self.learning_rate = learning_rate
        self.accumulated_sq_gradient = 0

    def update_weights(self, w, dEdW):
        """
        Updates the parameters.

        Parameters
        ----------
        w : numpy.ndarray
            The weight to be updated.
        dEdW : numpy.ndarray
            The derivative of error with respect to weight.
        """

        self.accumulated_sq_gradient += np.square(dEdW)
        return w - self.learning_rate * dEdW / (np.sqrt(self.accumulated_sq_gradient) + 1e-8)
