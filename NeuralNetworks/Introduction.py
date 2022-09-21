# -*- coding: utf-8 -*-

import numpy as np


def sigmoid(x):
    # activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Weight inputs, add bias, then use the activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


if __name__ == '__main__':
    print("An Introduction to Neural Networks!")

    weights = np.array([0, 1])  # w1 = 0, w2 = 1
    bias = 0  # b = 4
    n = Neuron(weights, bias)

    x = np.array([2, 3])
    x1 = n.feedforward(x)
    print(x1)

    x = np.array([x1, x1])
    print(n.feedforward(x))
