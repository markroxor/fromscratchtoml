#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under The MIT License - https://opensource.org/licenses/MIT

import tempfile
import os

pwd_path = os.path.dirname(__file__)


def _tempfile(fname):
    return os.path.join(tempfile.gettempdir(), fname)


def _test_data_path(fname):
    return os.path.join(pwd_path, "test_data", fname)
