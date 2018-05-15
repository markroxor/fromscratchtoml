#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# optimizes/updates the weights
class StochasticGradientDescent(object):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_weights(self, w, grad_wrt_w):
        return w - self.learning_rate * grad_wrt_w
