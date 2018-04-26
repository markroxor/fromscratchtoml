import torch as ch
import logging
import numpy as np

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Optimizer(object):

    def relu_activation(self, x):
        if x > 0:
            return 1
        return 0

    def sigmoid(self, x):
        return ch.exp(x) / (1 + ch.exp(x))

    def tanh(x):
        return (1 - ch.exp(x)) / (1 + ch.exp(x))

    def softmax(x):
        x_new = [ch.exp(i) for i in x]
        sum_x_new = sum(x_new)
        return [sum_x_new / (i) for i in x_new]

    def feed_forward(self, X_train, W_hidden, W_output):

        Weighted_X = X_train * W_hidden
        H_o = self.sigmoid(Weighted_X)

        Weighted_H_o = H_o * W_output
        Y_pred = self.relu_activation(Weighted_H_o)
        return Y_pred, H_o

    def backprop(self, X_train, Y_train, W_hidden, W_output, lr):
        Y_pred, H_o = self.feed_forward(X_train, W_hidden, W_output)

        # Layer Error
        Err_output = (Y_pred - Y_train) * self.relu_activation(W_output * H_o)
        Err_hidden = Err_output * W_output * self.relu_activation(H_o)

        # Derivatives with respect to the weights
        derivative_W_output = Err_output * H_o
        derivative_W_hidden = Err_hidden * X_train

        # Update the weights
        W_hidden -= lr * derivative_W_hidden
        W_output -= lr * derivative_W_output

    def SGD(self, X_train, Y_train, batch_size, lr):
        N = len(X_train)
        train_data = ch.utils.data.TensorDataset(X_train, Y_train)
        np.random.shuffle(train_data)
        mini_batches = np.array([train_data[i:i + batch_size]])
        for i in range([(0, N, batch_size)]):
            for X_train, Y_train in mini_batches:
                self.backprop(X_train, Y_train, lr)
