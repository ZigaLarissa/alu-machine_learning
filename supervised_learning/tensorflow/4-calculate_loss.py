#!/usr/bin/env python3
"""
Calculate the loss of a prediction.
"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the loss of a prediction.

    Args:
        y (tf.placeholder): The labels of the input data.
        y_pred (tf.Tensor): The predicted labels.

    Returns:
        tf.Tensor: The loss of the prediction.
    """

    # Calculate the loss.
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    return loss
