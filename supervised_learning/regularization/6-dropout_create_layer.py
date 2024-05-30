#!/usr/bin/env python3
"""
Create a layer of neural network
with dropout regularization
"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a TensorFlow layer with dropout regularization.

    Parameters:
    prev -- tensor containing the output of the previous layer
    n -- number of nodes the new layer should contain
    activation -- activation function that should be used on the layer
    keep_prob -- probability that a node will be kept

    Returns:
    The output of the new layer
    """
    # Create a dense layer
    dense_layer = tf.keras.layers.Dense(units=n, activation=activation)

    # Apply dropout to the dense layer's output
    dropout_layer = tf.keras.layers.Dropout(rate=1 - keep_prob)

    # Get the output of the dense layer
    dense_output = dense_layer(prev)

    # Apply dropout
    output = dropout_layer(dense_output)

    return output
