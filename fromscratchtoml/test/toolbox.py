#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Copyright (C) 2017 Dikshant Gupta <dikshant2210@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import tempfile
import os

import torch as ch
pwd_path = os.path.dirname(__file__)


def _tempfile(fname):
    return os.path.join(tempfile.gettempdir(), fname)


def _test_data_path(fname):
    return os.path.join(pwd_path, "test_data", fname)


def torch_equal(x, y, precision=1e-4):
    return not ch.sum(ch.ge(ch.abs(x - y), precision))
