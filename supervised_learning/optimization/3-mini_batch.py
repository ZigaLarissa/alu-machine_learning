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
    trains a loaded nn model using mini batch
    gradient descent

    args:
        X_train: np.ndarray (m, 784) containing the training data
        m: number of data points
        784: number of features
        Y_train: np.ndarray (m, 10) containing the training labels
        10: number of classes
        X_valid: np.ndarray (m, 784) containing the validation data
        Y_valid: np.ndarray (m, 10) containing the validation labels
        batch_size: number of data points in a batch
        epochs: number of times the training should pass through the whole dataset
        load_path: path from which to load the model
        save_path: path to save the model

    returns:
        the path where the model was saved
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        # Retrieve the necessary tensors and operations from the graph
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        y = graph.get_tensor_by_name('y:0')
        accuracy = graph.get_tensor_by_name('accuracy:0')
        loss = graph.get_tensor_by_name('loss:0')
        train_op = graph.get_operation_by_name('train_op')

        # Loop over epochs
        for epoch in range(epochs):
            # Shuffle the training data
            X_train, Y_train = shuffle_data(X_train, Y_train)

            m = X_train.shape[0]
            # Loop over batches

            # print the metrics after each epoch
            train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            
            print(f"After {epoch + 1} epochs:")
            print(f"\tTraining Cost: {train_cost}")
            print(f"\tTraining Accuracy: {train_accuracy}")
            print(f"\tValidation Cost: {valid_cost}")
            print(f"\tValidation Accuracy: {valid_accuracy}")

            for i in range(0, m, batch_size):
                end = i + batch_size
                X_batch = X_train[i:end]
                Y_batch = Y_train[i:end]

                # Train the model on the current batch
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                # Print step metrics every 100 steps
                if i % 100 == 0:
                    step_cost, step_accuracy = sess.run([loss, accuracy], feed_dict={x: X_batch, y: Y_batch})
                    print(f"\tStep {i}:")
                    print(f"\t\tCost: {step_cost}")
                    print(f"\t\tAccuracy: {step_accuracy}")

        # Save the trained model
        saver.save(sess, save_path)

    return save_path
