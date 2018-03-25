#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Copyright (C) 2017 Dikshant Gupta <dikshantgupta2210@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import torch as ch
from omega.toolbox import sigmoid, deriv_sigmoid


class NetworkMesh(object):
    def __init__(self, layer_architecture):
        self.layer_architecture = layer_architecture
        self.num_layers = len(layer_architecture)
        self.layerwise_biases = [ch.randn(1, x) for x in layer_architecture[1:]]
        self.layerwise_weights = [ch.randn(x, y) for x, y in zip(layer_architecture[:-1], layer_architecture[1:])]

    def feedforward(self, x):
        x = x.view(1, ch.numel(x))
        for biases, weights in zip(self.layerwise_biases, self.layerwise_weights):
            x = sigmoid(ch.mm(x, weights) + biases)
        return x

    def SGD(self, train_data, epochs, batch_size, eta, test_data):
        for i in range(epochs):
            batches = ch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
            for batch in batches:
                self.update_batch(batch, eta)

            print("Epoch {0}: {1}/{2}".format(i, self.evaluate(test_data), len(test_data)))

    def update_batch(self, batch, eta):
        nabla_b = [ch.zeros(biases.size()) for biases in self.layerwise_biases]
        nabla_w = [ch.zeros(weights.size()) for weights in self.layerwise_weights]

        for x, y in zip(batch[0], batch[1]):
            x = x.view(1, ch.numel(x))
            if isinstance(y, int):
                _y = ch.zeros(self.layer_architecture[-1])
                _y[y] = 1
                y = _y

            y = y.view(1, ch.numel(y))
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [ch.add(nb, dnb) for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [ch.add(nw, dnw) for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.update_weights(eta, nabla_b, nabla_w, len(batch))

    def update_weights(self, eta, nabla_b, nabla_w, batch_size):
        self.layerwise_biases = [b - (eta) * nb
                                 for b, nb in zip(self.layerwise_biases, nabla_b)]
        self.layerwise_weights = [w - (eta) * nw
                                 for w, nw in zip(self.layerwise_weights, nabla_w)]

    def backprop(self, x, y):
        nabla_b = [ch.zeros(biases.size()) for biases in self.layerwise_biases]
        nabla_w = [ch.zeros(weights.size()) for weights in self.layerwise_weights]

        activation = x
        activations = [activation]
        zs = []
        for biases, weights in zip(self.layerwise_biases, self.layerwise_weights):
            z = ch.mm(activation, weights) + biases
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * deriv_sigmoid(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = ch.mm(activations[-2].transpose(0, 1), delta)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = deriv_sigmoid(z)
            delta = ch.mm(delta, self.layerwise_weights[-l + 1].transpose(0, 1)) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = ch.mm(activations[-l - 1].transpose(0, 1), delta)
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        correct_evaluation = 0
        for d in test_data:
            X, Y = d
            X = X.view(1, ch.numel(X))
            if isinstance(Y, int):
                _y = ch.zeros(self.layer_architecture[-1])
                _y[Y] = 1
                Y = _y
            Y = Y.view(1, ch.numel(Y))
            a = self.feedforward(X)

            _, prediction = ch.max(a, 1)
            _, target = ch.max(Y, 1)
            if (prediction == target).numpy():
                correct_evaluation += 1

        return correct_evaluation

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
