#!/usr/bin/env python3
"""
Creates a layer that includes L2 regularization
usinf TensorFlow
"""


import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a TensorFlow layer that includes L2 regularization.

    Parameters:
    prev -- tensor containing the output of the previous layer
    n -- number of nodes the new layer should contain
    activation -- activation function that should be used on the layer
    lambtha -- L2 regularization parameter
    
    Returns:
    The output of the new layer
    """
    regularizer = tf.keras.regularizers.L2(l2=lambtha)
    layer = tf.keras.layers.Dense(units=n, activation=activation, kernel_regularizer=regularizer)
    return layer(prev)
