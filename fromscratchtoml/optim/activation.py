import torch as ch
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Activation(object):

    @staticmethod
    def relu_activation(x):
        return max(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + ch.exp(-x))

    @staticmethod
    def tanh(x):
        return (1 - ch.exp(x)) / (1 + ch.exp(x))

    @staticmethod
    def softmax(x):
        x_new = [ch.exp(i) for i in x]
        sum_x_new = sum(x_new)
        return [sum_x_new / (i) for i in x_new]

    @staticmethod
    def derivate_relu(x):
        if x > 0:
            return 1
        return 0

    @staticmethod
    def derivate_sigmoid(x):
        return (Activation.sigmoid(x)) * (1 - Activation.sigmoid(x))

    @staticmethod
    def derivate_tanh(x):
        return - ch.exp(x) / (1 + ch.exp(x)) ** 2
