
import torch as ch
import logging
import numpy as np
from activation import Activation
from loss_funtion import LossFunction

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Optimizer(object):

    def __init__(self, no_hidden_layer_neurons, no_output_layer_neurons, X_train, Y_train, lr=0.001,
                bias_h=0, epoch=500):
                self.no_hidden_layer_neurons = no_hidden_layer_neurons
                self.no_output_layer_neurons = no_output_layer_neurons
                self.X_train = X_train
                self.Y_train = Y_train
                self.bias_h = bias_h
                self.epoch = epoch
                shape_xtrain = X_train.size()
                shape_weight_hidden = (shape_xtrain[0], no_hidden_layer_neurons)
                np.random.seed(42)
                # Weight, bias initialization
                self.weight_hidden = np.random.uniform(size=shape_weight_hidden)
                self.bias_hidden = np.random.uniform(size=(1, no_hidden_layer_neurons))
                self.weight_output = np.random.uniform(size=(no_hidden_layer_neurons, no_output_layer_neurons))
                self.bias_output = np.random.uniform(size=(1, no_output_layer_neurons))

    def feed_forward(self):

                original_input = np.dot(self.X_train, self.weight_hidden)
                input_to_hidden_layer = original_input + self.bias_hidden
                output_after_hidden_activation = Activation.sigmoid(input_to_hidden_layer)
                original_input_to_output_layer = np.dot(output_after_hidden_activation, self.weight_output)
                input_to_output_layer = original_input_to_output_layer + self.bias_output
                final_output = Activation.sigmoid(input_to_output_layer)
                return output_after_hidden_activation, final_output

    def backprop(self):
                output_after_hidden_activation, final_output = self.feed_forward()
                error_output = LossFunction.L2_loss(final_output, self.Y_train)
                slope_output_layer = Activation.derivate_sigmoid(error_output)
                slope_hidden_layer = Activation.derivate_sigmoid(output_after_hidden_activation)
                d_output = error_output * slope_output_layer
                Error_at_hidden_layer = d_output.dot(self.weight_output.T)
                d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
                self.weight_output += (output_after_hidden_activation.T).dot(d_output) * self.lr
                self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.lr
                self.weight_hidden += (self.X_train.T).dot(d_hiddenlayer) * self.lr
                self.bias_hidden += np.sum(d_hiddenlayer, axis=0, keepdims=True) * self.lr
                return final_output

    def SGD(self, batch_size):
        N = len(self.X_train)
        train_data = ch.utils.data.TensorDataset(self.X_train, self.Y_train)
        np.random.shuffle(train_data)
        for i in range([(0, N, batch_size)]):
            mini_batches = np.array([train_data[i:i + batch_size]])
            for X_train, Y_train in mini_batches:
                logger.info("The predicted output {} at epoch {}").format(self.backprop(), self.epoch)
