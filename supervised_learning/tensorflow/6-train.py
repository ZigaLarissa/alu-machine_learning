#!/usr/bin/env python3
"""
Train a neural network.
"""


import tensorflow as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Trains a neural network.

    Args:
        X_train (np.ndarray): The training input data.
        Y_train (np.ndarray): The training labels.
        X_valid (np.ndarray): The validation input data.
        Y_valid (np.ndarray): The validation labels.
        layer_sizes (list): A list containing the number of nodes in each
        layer of the network.
        activations (list): A list containing the activation functions for
        each layer of the network.
        alpha (float): The learning rate.

    Returns:
        str: The path where the model was saved.
    """
    # Reset the graph.
    tf.reset_default_graph()

    # Create placeholders.
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Forward propagation.
    y_pred = forward_prop(x, layer_sizes, activations)

    # Accuracy and loss.
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)

    # Training operation.
    train_op = create_train_op(loss, alpha)

    # add variables to collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)


    # Saver.
    saver = tf.train.Saver()

    # Create a session.
    with tf.Session() as sess:
        # Initialize all variables.
        sess.run(tf.global_variables_initializer(mode='FAN_AVG'))

        # Train the network.
        for i in range(iterations + 1):
            # Train the network.
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})

            # Print the metrics.
            if i % 100 == 0 or i == 0 or i == iterations:
                train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
                train_accuracy = sess.run(
                    accuracy, feed_dict={x: X_train, y: Y_train}
                    )
                valid_cost = sess.run(
                    loss, feed_dict={x: X_valid, y: Y_valid}
                    )
                valid_accuracy = sess.run(
                    accuracy, feed_dict={x: X_valid, y: Y_valid}
                    )

                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))

        # Save the model.
        save_path = saver.save(sess, save_path)

    return save_path
