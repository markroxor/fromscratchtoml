#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under The MIT License - https://opensource.org/licenses/MIT

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

    # return np.array(a)
    return a
