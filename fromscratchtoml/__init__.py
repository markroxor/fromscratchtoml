#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under The MIT License - https://opensource.org/licenses/MIT

import logging
import numpy as np  # noqa:F401

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def use(backend):
    global np
    if backend == "numpy":
        import numpy as np   # noqa:F401
        logging.debug("Using numpy backend.")
    elif backend == "cupy":
        import cupy as np
        logging.debug("Using cupy backend.")
    else:
        raise ImportError('Only available backends are cupy or numpy.')
