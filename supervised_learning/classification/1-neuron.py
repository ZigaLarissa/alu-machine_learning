#!/usr/bin/env python3

"""
this module contains a class called Neuron
"""
import numpy as np


class Neuron:
    """Neuron class defines a single neuron
    performing binary classification"""

    def __init__(self, nx):
        """
        Constructor for the Neuron class.

        Args:
            nx (int): The number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.

        Private Attributes:
            W: The weights vector for the neuron.
            b: The bias for the neuron.
            A: The activated output of the neuron (prediction).
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter fuction for the Weight vector"""
        return self.__W

    @property
    def b(self):
        """Getter fuction for the bias"""
        return self.__b

    @property
    def A(self):
        """Getter fuction for the activated output"""
        return self.__A
