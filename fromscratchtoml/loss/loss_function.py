import torch as ch
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LossFunction(object):

    @staticmethod
    def cross_entropy(Y_pred, Y_train):
        if Y_pred == 1:
            return -ch.log(Y_train)
        else:
            return -ch.log(1 - Y_train)

    @staticmethod
    def hinge_loss(Y_pred, Y_train):
        return ch.max(0, 1 - Y_pred * Y_train)

    @staticmethod
    def L1_loss(Y_pred, Y_train):
        return ch.sum(ch.absolute(Y_pred - Y_train))

    @staticmethod
    def L2_loss(Y_pred, Y_train):
        return ch.sum(ch.pow((Y_pred - Y_train), 2)) / len(Y_train)
