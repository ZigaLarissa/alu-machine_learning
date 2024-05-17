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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X (ndarray): An array with shape (nx, m) that contains the input
                         data.

        Returns:
            ndarray: An array with shape (1, m) containing the activated output
                     of the neural network and the cache,
                     a dictionary containing the activation of each layer.
        """
        # calculate the nodes input
        Z1 = np.matmul(self.W1, X) + self.b1

        # calculate the activation of the nodes
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # calculate the output
        Z2 = np.matmul(self.W2, self.A1) + self.b2

        # calculate the activation of the output
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.A1, self.A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
            Y (ndarray): An array with shape (1, m) with the correct labels
                         that classify if the data is 0 or 1.
            A (ndarray): An array with shape (1, m) containing the activated
                         output of the neuron for each example.

        Returns:
            float: The cost of the model.
        """
        # number of examples
        m = Y.shape[1]

        # calculate the cost
        cost = -np.sum((Y * np.log(A)) + ((1-Y) * np.log(1.0000001 - A))) / m

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions.
        """

        # calculate the forward propagation
        self.forward_prop(X)

        # calculate the cost
        cost = self.cost(Y, self.A2)

        # calculate the prediction
        prediction = np.where(self.A2 >= 0.5, 1, 0)

        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.
        """
        # number of examples
        m = Y.shape[1]

        # calculate the gradient of the output layer
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        # calculate the gradient of the hidden layer
        dz1 = np.matmul(self.W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        # update the weights and biases
        self.__W2 = self.W2 - (alpha * dw2)
        self.__b2 = self.b2 - (alpha * db2)
        self.__W1 = self.W1 - (alpha * dw1)
        self.__b1 = self.b1 - (alpha * db1)

        return self.W1, self.b1, self.W2, self.b2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network.
        """
        # number of examples
        m = Y.shape[1]

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        # start the training
        cost = []

        for i in range(iterations):
            # calculate the forward propagation
            self.forward_prop(X)

            # calculate the cost
            c = self.cost(Y, self.A2)
            cost.append(c)

            # calculate the gradient descent
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)

        # evaluate the training
        evaluation = self.evaluate(X, Y)
        return evaluation
