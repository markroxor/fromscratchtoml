#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np

def use(backend):
    global np
    if backend == "numpy":
        import numpy as np
        logging.debug("Using numpy backend.")
    elif backend == "cupy":
        import cupy as np
        logging.debug("Using cupy backend.")
    else:
        raise ImportError('Only available backends are cupy or numpy.')
