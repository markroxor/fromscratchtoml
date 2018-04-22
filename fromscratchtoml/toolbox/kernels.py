#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import torch as ch
import math


class Kernel(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def linear(self):
        def fun(x, y):
            return ch.dot(x, y)
        return fun

    def polynomial(self):
        def fun(x, y):
            return pow((ch.dot(x, y) + self.kwargs['const']), self.kwargs['degree'])
        return fun

    def rbf(self):
        def fun(x, y):
            euclidean_dist = pow(ch.norm(x - y), 2)
            return math.exp(-self.kwargs['gamma'] * euclidean_dist)
        return fun
