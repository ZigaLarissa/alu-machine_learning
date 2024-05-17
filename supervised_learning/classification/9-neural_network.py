#!/usr/bin/env python3v
"""
This module contains the NeuralNetwork class.
"""
import numpy as np


class NeuralNetwork:
    """
    Class NeuralNetwork: defines a neural network with one
    hidden layer performing binary classification.
    """
    def __init__(self, nx, nodes):
        """
        Constructor of the NeuralNetwork class.

        Args:
            nx (int): The number of input features.
            nodes (int): The number of nodes found in the hidden layer.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If nodes is not an integer.
            ValueError: If nodes is less than 1.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initializes the normal attributes of the class
        self.nx = nx
        self.nodes = nodes

        """Initialize weights and biases of the hidden layer
        as private attributes"""
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        """Initialize weights and biases of the output neuron
        as private attributes"""
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    # Getters
    @property
    def W1(self):
        """
        Getter method that returns the value of the attribute "W1".
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter method that returns the value of the attribute "b1".
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter method that returns the value of the attribute "A1".
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter method that returns the value of the attribute "W2".
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter method that returns the value of the attribute "b2".
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter method that returns the value of the attribute "A2".
        """
        return self.__A2
