#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

from functools import partial

from .. import Activations

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Activation(object):
    def __init__(self, activation=None, trainable=False):
        self.activation = partial(getattr(Activations, activation))

    def forward(self, X, return_deriv=False):
        self.input = X
        self.output, self.output_deriv = self.activation(X, return_deriv=True)

        if return_deriv:
            return self.output, self.output_deriv

        return self.output

    def back_propogate(self, delta):
        delta = delta * self.output_deriv
        return delta, 0, 0

    def optimize(self, optimizer, der_cost_bias, der_cost_weight):
        # This is a non trainable layer
        return
