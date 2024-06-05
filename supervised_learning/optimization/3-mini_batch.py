#!/usr/bin/env python3
"""
trains a loaded neural network model using
mini-batch gradient descent
"""

import tensorflow as tf


shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt", 
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent.

    Args:
        X_train (np.ndarray): Training data of shape (m, 784).
        Y_train (np.ndarray): One-hot training labels of shape (m, 10).
        X_valid (np.ndarray): Validation data of shape (m, 784).
        Y_valid (np.ndarray): One-hot validation labels of shape (m, 10).
        batch_size (int): Number of data points in a batch.
        epochs (int): Number of times the training should pass through the whole dataset.
        load_path (str): Path from which to load the model.
        save_path (str): Path to where the model should be saved after training.

    Returns:
        str: Path where the model was saved.
    """
    with tf.Session() as sess:
        # import meta graph and restore weights
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        # get ops and placeholders
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        for epoch in range(epochs +1):
            # shuffle the data
            X_train_shuffled, Y_train_shuffled = shuffle_data(X_train, Y_train)

            # mini-batch
            steps = ( X_train_shuffled.shape[0] // batch_size ) + 1

            # Get the number of minibatches
            steps = (X_train_shuffled.shape[0] // batch_size) + 1

            # Loop over the batches
            for step in range(steps):
                start = step * batch_size
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                Y_batch = Y_train_shuffled[start:end]

                # Train the model
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if step % 100 == 0:
                    # Calculate the cost and accuracy on the current mini-batch
                    step_cost = sess.run(loss, feed_dict={x: X_batch, y: Y_batch})
                    step_accuracy = sess.run(accuracy, feed_dict={x: X_batch, y: Y_batch})
                    print(f"\tStep {step}:")
                    print(f"\t\tCost: {step_cost}")
                    print(f"\t\tAccuracy: {step_accuracy}")

            # Calculate the cost and accuracy on the entire training and validation sets
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            print(f"After {epoch} epochs:")
            print(f"\tTraining Cost: {train_cost}")
            print(f"\tTraining Accuracy: {train_accuracy}")
            print(f"\tValidation Cost: {valid_cost}")
            print(f"\tValidation Accuracy: {valid_accuracy}")

        # Save the session
        saver.save(sess, save_path)

    return save_path
