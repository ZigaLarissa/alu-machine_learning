#!/usr/bin/env python3v
"""
This module contains the DeepNeuralNetwork class.
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Class DeepNeuralNetwork: defines a deep neural network
    performing binary classification.
    """
    def __init__(self, nx, layers):
        """
        Constructor of the DeepNeuralNetwork class.

        Args:
            nx (int): The number of input features.
            layers (list): List representing the number of nodes in
            each layer of the network.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If layers is not a list.
            ValueError: If layers does not represent a list
            of positive integers.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # Public attributes
        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # weights instantiation
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.weights["W1"] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.weights["W{}".format(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

            # Initialize biases
            self.weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
