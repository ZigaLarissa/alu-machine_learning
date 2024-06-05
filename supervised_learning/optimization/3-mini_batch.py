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
        sever = tf.train.import_meta_graph(load_path + '.meta')
        sever.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection("train_op")[0]

        m = X_train.shape[0]
        if m % batch_size == 0:
            num_batches = m // batch_size
        else:
            num_batches = m // batch_size + 1

        for epoch in range(epochs + 1):
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

            loss_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            acc_train = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            loss_valid = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            acc_valid = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(loss_train))
            print("\tTraining Accuracy: {}".format(acc_train))
            print("\tValidation Cost: {}".format(loss_valid))

        save = sever.save(sess, save_path)
    return save
