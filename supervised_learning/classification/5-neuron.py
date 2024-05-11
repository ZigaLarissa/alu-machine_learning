#!/usr/bin/env python3
"""
This module contains the Neuron class.
"""
import numpy as np


class Neuron:
    """ Class Neuron: defines a single neuron
    performing binary classification. """

    def __init__(self, nx):
        """
        Constructor for the Neuron class.

        Args:
            nx (int): The number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
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
        """W getter function"""
        return self.__W

    @property
    def b(self):
        """b getter function"""
        return self.__b

    @property
    def A(self):
        """A getter function"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The activated output.
        """
        # define the neuron's linear combination function
        z = np.matmul(self.__W, X) + self.__b
        # define the neuron's activation function
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost(how good our prediction model is)
        of the model using logistic regression.

        Args:
            Y (np.ndarray): The correct labels.
            A (np.ndarray): The activated output/predictions.

        Returns:
            float: The cost.
        """
        # number of examples
        m = Y.shape[1]
        # compute the cost
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.

        Args:
            Y(np.ndarray): The correct labels.
            X(np.ndarray): The input data.

        Returns:
            np.ndarray, float: The neuron's prediction and
            the cost of the network, respectively.
        """
        # calculate the neuron's prediction
        A = self.forward_prop(X)
        # convert the neuron's prediction to 1 or 0 (binary classification)
        prediction = np.where(A >= 0.5, 1, 0)
        # calculate the cost
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.

        Args:
            X (np.ndarray): The input data.
            Y (np.ndarray): The correct labels.
            A (np.ndarray): The activated output/predictions.
            alpha (float): The learning rate.

        Returns:
            None
        """
        # number of examples
        m = Y.shape[1]
        # calculate the gradient
        error = A - Y
        dw = 1 / m * np.matmul(X, error.T)
        db = 1 / m * np.sum(error)
        # update the neuron's weights and bias
        self.__W -= alpha * dw.T
        self.__b -= alpha * db
        return None
