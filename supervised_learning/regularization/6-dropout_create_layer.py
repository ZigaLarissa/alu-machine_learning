#!/usr/bin/env python3
"""
Create a layer of neural network
with dropout regularization
"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
        Creates a new layer in a neural network with dropout regularization.

        Args:
            prev (tf.Tensor): Output tensor from the previous layer.
            n (int): Number of nodes in the new layer.
            activation (callable): Activation function to apply to the new layer.
            keep_prob (float): Probability of keeping a node during dropout.

        Returns:
            tf.Tensor: Output tensor of the new layer.
    """
    # Initialize weights and biases
    init = tf.contrib.layers.variance_scaling_initializer(
      mode="FAN_AVG"
    )
    weights = tf.Variable(
        init([int(prev.get_shape()[1]), n]),
        name="weights"
    )
    biases = tf.Variable(
        tf.zeros(n),
        name="biases"
    )
    # Dropout layer
    dropout = tf.nn.dropout(prev, keep_prob)
    # Linear combination
    z = tf.matmul(dropout, weights) + biases
    # Activation function
    return activation(z)
