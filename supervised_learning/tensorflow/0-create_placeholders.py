#!/usr/bin/env python3
"""Create placeholders for input data and labels"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Create placeholders for input data and labels

    Args:
        nx (int): The number of feature columns in our data.
        classes (int): The number of classes in our classifier.

    Returns:
        tuple: (x, y)
            x (tf.placeholder): The input data placeholder.
            y (tf.placeholder): The labels placeholder.
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    return x, y
