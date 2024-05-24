#!/usr/bin/env python3
"""
Calculate the accuracy of a prediction.
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Args:
        y (tf.placeholder): The labels of the input data.
        y_pred (tf.Tensor): The predicted labels.

    Returns:
        tf.Tensor: The accuracy of the prediction.
    """

    # Determine if the predictions are correct.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))

    # Calculate the accuracy.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
