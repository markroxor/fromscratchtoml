#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

from fromscratchtoml import np

import unittest
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Gradient_check(object):

    def function(self, h, **kwargs):
        pass

    def compute_relative_error(self, loss, **kwargs):
        # numerical solution
        self.h = 1e-7
        Ep, _ = self.function(self.h, loss, **kwargs)
        En, _ = self.function(-self.h, loss, **kwargs)
 
        dEdW = (np.sum(Ep - En)) / ( 2. * self.h)

        # analytic solution
        _, dEdW_ = self.function(0, loss, **kwargs)
        dEdW_ = dEdW_[0][0]

        relative_error = np.abs(dEdW - dEdW_) / max(np.abs(dEdW), np.abs(dEdW_))
        return relative_error