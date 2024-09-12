#!/usr/bin/env python3
"""
This module contains the GRUCell class.
"""

import numpy as np


class GRUCell:
    """Represents a Gated Recurrent Unit (GRU) cell."""

    def __init__(self, i, h, o):
        """
        Initializes the GRUCell.

        Parameters:
        i -- Dimensionality of the input data
        h -- Dimensionality of the hidden state
        o -- Dimensionality of the output
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, z):
        """Applies the sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        """Applies the softmax activation function."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step of the GRU cell.

        Parameters:
        h_prev -- numpy.ndarray of shape (m, h), containing the previous hidden state
        x_t -- numpy.ndarray of shape (m, i), containing the data input at time step t
        m -- Batch size
        i -- Dimensionality of the input
        h -- Dimensionality of the hidden state

        Returns:
        h_next -- The next hidden state
        y -- The output of the cell
        """
        m, _ = x_t.shape
        h = h_prev.shape[1]

        # Concatenate the previous hidden state and current input
        h_x_concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z_t = self.sigmoid(np.dot(h_x_concat, self.Wz) + self.bz)

        # Reset gate
        r_t = self.sigmoid(np.dot(h_x_concat, self.Wr) + self.br)

        # Candidate hidden state (intermediate hidden state)
        h_r_concat = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_hat = np.tanh(np.dot(h_r_concat, self.Wh) + self.bh)

        # Compute the next hidden state
        h_next = (1 - z_t) * h_prev + z_t * h_hat

        # Compute the output of the cell (softmax activation)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y
