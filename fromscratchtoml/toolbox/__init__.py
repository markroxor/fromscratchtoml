#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import torch as ch


def sigmoid(x):
    """Returns the sigmoid of x.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    sigmoid of x

    """
    return 1.0 / (1.0 + ch.exp(-x))


def deriv_sigmoid(x):
    """Returns the derivative of sigmoid of x.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    derivative of sigmoid of x

    """
    return sigmoid(x) * (1 - sigmoid(x))
