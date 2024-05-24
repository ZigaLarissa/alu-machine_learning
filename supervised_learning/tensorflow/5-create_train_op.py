#!/usr/bin/env python3
"""
Create the training operation for the network.
"""


import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network.

    Args:
        loss (tf.Tensor): The loss of the network's prediction.
        alpha (float): The learning rate.

    Returns:
        tf.Operation: The training operation.
    """

    # Create the optimizer.
    optimizer = tf.train.GradientDescentOptimizer(alpha)

    # Minimize the loss.
    train_op = optimizer.minimize(loss)

    return train_op
