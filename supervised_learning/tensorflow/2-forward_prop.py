#!/usr/bin/env python3
"""
Create forward propagation for our neural network"""


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Args:
        x (tf.placeholder): The input data.
        layer_sizes (list): A list containing the number of nodes in each
        layer of the network.
        activations (list): A list containing the activation functions for
        each layer of the network.

    Returns:
        tf.Tensor: The prediction of the network.
    """

    # Create the first layer.
    output = create_layer(x, layer_sizes[0], activations[0])

    # Create subsequent layers.
    for i in range(1, len(layer_sizes)):
        output = create_layer(output, layer_sizes[i], activations[i])

    return output
