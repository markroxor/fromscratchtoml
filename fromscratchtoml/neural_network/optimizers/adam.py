#!/usr/bin/env python
# -*- coding: utf-8 -*-
#



from fromscratchtoml import np

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# optimizes/updates the weights
class Adam(object):
    """
    The Adam optimizer.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, bias_fix=False):
        """
        Initialising the optimizer parameters.

        Parameters
        ----------
        learning_rate : float
            the rate of change of weights. The higher the learning rate - more is the change in the parameters.
        beta1 : float
            Beta1.
        beta2 : float
            Beta2.
        bias_fix : bool
            Fix bias, read paper for more details.
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.bias_fix = bias_fix

        self.accumulated_gradient = 0
        self.accumulated_sq_gradient = 0

        self.t = 1

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

        self.accumulated_gradient = self.beta1 * self.accumulated_gradient + (1 - self.beta1) * dEdW
        self.accumulated_sq_gradient = self.beta2 * self.accumulated_sq_gradient + (1 - self.beta2) * np.square(dEdW)

        if self.bias_fix:
            self.t += 1
            accumulated_gradient_t = self.accumulated_gradient / (1 - (self.beta1 ** self.t))
            accumulated_sq_gradient_t = self.accumulated_sq_gradient / (1 - (self.beta2 ** self.t))
            return w - self.learning_rate * accumulated_gradient_t / (np.sqrt(accumulated_sq_gradient_t) + 1e-8)

        return w - self.learning_rate * self.accumulated_gradient / (np.sqrt(self.accumulated_sq_gradient) + 1e-8)
