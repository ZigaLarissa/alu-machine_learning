#!/usr/bin/env python3
"""
Evaluate the output of a neural network.
"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Args:
        X (np.ndarray): The input data.
        Y (np.ndarray): The correct labels.
        save_path (str): The location to load the model from.

    Returns:
        np.ndarray: The network's prediction, or None on failure.
    """

    # Load the model.
    with tf.Session() as session:
        loader = tf.train.import_meta_graph(save_path + '.meta')
        loader.restore(session, save_path)

        # Get the placeholders from the model.
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]

        # Get the prediction.
        y_pred = tf.get_collection('y_pred')[0]

        # Get the accuracy.
        accuracy = tf.get_collection('accuracy')[0]

        # Get the loss.
        loss = tf.get_collection('loss')[0]

        # Evaluate the model.
        prediction = session.run(y_pred, feed_dict={x: X, y: Y})
        acc = session.run(accuracy, feed_dict={x: X, y: Y})
        loss = session.run(loss, feed_dict={x: X, y: Y})

    return prediction, acc, loss
