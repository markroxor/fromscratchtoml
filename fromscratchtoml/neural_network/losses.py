#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import numpy as np

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def mean_squared_error(y_predicted, y_target, return_deriv=False):
    if return_deriv:
        return np.mean(np.square(y_predicted - y_target)), y_predicted - y_target
    return np.mean(np.square(y_predicted - y_target))
