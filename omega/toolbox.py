#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Copyright (C) 2017 Dikshant Gupta <dikshant2210@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import torch as ch


def sigmoid(x):
    return 1.0 / (1.0 + ch.exp(-x))


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
