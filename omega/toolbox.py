import torch as ch


def sigmoid(x):
    return 1.0 / (1.0 + ch.exp(-x))


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
