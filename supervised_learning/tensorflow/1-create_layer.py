#!/usr/bin/env python3
"""Create layers for our neural network"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Create a layer in a neural network.

    Args:
        prev (tf.Tensor): The tensor output of the previous layer.
        n (int): The number of nodes in the layer to create.
        activation (tf.nn.activation_function): The activation function to use for the layer.

    Returns:
        tf.Tensor: The tensor output of the created layer.
    """
    with tf.variable_scope("layer"):
        # Use He et. al initialization for the layer weights
        initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

        # Create the weights and biases for the layer
        weights = tf.get_variable("weights", shape=[prev.shape[1], n], initializer=initializer)
        biases = tf.get_variable("biases", shape=[n], initializer=tf.zeros_initializer())

        # Compute the layer output
        layer_output = tf.matmul(prev, weights) + biases

        # Apply the activation function
        if activation is not None:
            layer_output = activation(layer_output)

    return layer_output
