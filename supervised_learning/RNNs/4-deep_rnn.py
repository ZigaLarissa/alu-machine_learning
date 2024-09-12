#!/usr/bin/env python3
"""
This module contains the DeepRNN class.
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Parameters:
    rnn_cells -- List of RNNCell instances representing each layer of the RNN
    X -- numpy.ndarray of shape (t, m, i) containing the input data, where
          t is the maximum number of time steps,
          m is the batch size,
          i is the dimensionality of the input data
    h_0 -- numpy.ndarray of shape (l, m, h) containing the initial hidden
    state for each layer, where
           l is the number of layers,
           m is the batch size,
           h is the dimensionality of the hidden state

    Returns:
    H -- numpy.ndarray of shape (t, l, m, h) containing all of the hidden
    states for all time steps and layers
    Y -- numpy.ndarray of shape (t, m, o) containing all of the outputs
    for all time steps
    """
    l = len(rnn_cells)  # Number of layers
    t, m, i = X.shape

    # Check dimensional consistency
    if h_0.shape[0] != l or h_0.shape[1] != m or h_0.shape[2] != h:
        raise ValueError(
            "Initial hidden state h_0 should have the shape (l, m, h)"
            )

    H = np.zeros((t, l, m, rnn_cells[0].Wh.shape[1]))
    Y = np.zeros((t, m, rnn_cells[-1].Wy.shape[1]))

    # Initialize the hidden state for the first layer
    h_prev = h_0[0]

    for time_step in range(t):
        x_t = X[time_step]

        # Propagate through each layer
        for layer in range(l):
            if layer == 0:
                # Input to the first layer
                h_next, _ = rnn_cells[layer].forward(h_prev, x_t)
            else:
                # Subsequent layers
                h_next, _ = rnn_cells[layer].forward(h_prev, x_t)

            H[time_step, layer] = h_next

            # Update the hidden state for the next layer
            h_prev = h_next

        # Output from the last layer
        _, y = rnn_cells[-1].forward(h_prev, x_t)
        Y[time_step] = y  # Store output for the current time step

    return H, Y
