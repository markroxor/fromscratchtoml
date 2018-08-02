#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

from fromscratchtoml import np
import numpy

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def to_onehot(y):

    try:
        import cupy
        if isinstance(y, cupy.core.core.ndarray):
            y = np.asnumpy(y)
    except ImportError:
        pass

    unq, _ = numpy.unique(y, return_inverse=True)

    # a = []
    a = np.zeros((len(y), len(unq)))
    for i in range(len(y)):
        # x = np.zeros(len(unq))
        a[i][int(y[i])] = 1.
        # a.append(x)

    return a


def vocab_one_hot(Y, vocab_size):
    temp = np.zeros((Y.shape[0], Y.shape[1], vocab_size))

    for i, _ in enumerate(Y):
        temp[i] = np.eye(vocab_size)[Y[i]]

    return temp
