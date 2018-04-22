#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import torch as ch
import math


def linear(x, y):
    return ch.dot(x, y)


def polynomial(x, y, const=0, degree=1):
    return pow((ch.dot(x, y) + const), degree)


def rbf(x, y, gamma=0.1):
    euclidean_dist = pow(ch.norm(x - y), 2)
    return math.exp(-gamma * euclidean_dist)
