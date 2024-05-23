#!/usr/bin/env python3
"""Create layers for our neural network"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a fully connected layer with He et al. initialization.

    Args:
        prev (tf.Tensor): The tensor output of the previous layer.
        n (int): The number of nodes in the layer to create.
        activation (str or callable): The activation function
        to use for the layer.

    Returns:
        tf.Tensor: The tensor output of the created layer.
    """

    # He et al. initialization for the layer weights
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")

    # Create the layer with tf.layers.Dense.
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name='layer')

    return layer(prev)
