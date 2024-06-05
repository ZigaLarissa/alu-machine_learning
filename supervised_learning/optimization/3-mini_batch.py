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
            for i in range(0, m, batch_size):
                X_batch = X_train[i:i + batch_size]
                Y_batch = Y_train[i:i + batch_size]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                # Print the cost and accuracy of the batch
                if i % 100 is 0:
                    cost = sess.run(loss, feed_dict={x: X_batch, y: Y_batch})
                    acc = sess.run(accuracy, feed_dict={x: X_batch, y: Y_batch})
                    print("\tStep {}:".format(i))
                    print("\t\tCost: {}".format(cost))
                    print("\t\tAccuracy: {}".format(acc))

            # Calculate the cost and accuracy of the epoch
            loss_train = sess.run(loss,
                                  feed_dict={x: X_train, y: Y_train})
            accuracy_train = sess.run(accuracy,
                                      feed_dict={x: X_train, y: Y_train})
            loss_valid = sess.run(loss,
                                  feed_dict={x: X_valid, y: Y_valid})
            accuracy_valid = sess.run(accuracy,
                                      feed_dict={x: X_valid, y: Y_valid})
            

        # print metrics
        print("After {} epochs:".format(epoch))
        print("\tTraining Cost: {}".format(loss_train))
        print("\tTraining Accuracy: {}".format(accuracy_train))
        print("\tValidation Cost: {}".format(loss_valid))
        print("\tValidation Accuracy: {}".format(accuracy_valid))
                
        epoch += 1
        # Calculate the cost and accuracy of the epoch
        loss_train = sess.run(loss,
                                feed_dict={x: X_train, y: Y_train})
        accuracy_train = sess.run(accuracy,
                                    feed_dict={x: X_train, y: Y_train})
        loss_valid = sess.run(loss,
                                feed_dict={x: X_valid, y: Y_valid})
        accuracy_valid = sess.run(accuracy,
                                    feed_dict={x: X_valid, y: Y_valid})
        
        # print metrics
        print("After {} iterations:".format(i))
        print("\tTraining Cost: {}".format(loss_train))
        print("\tTraining Accuracy: {}".format(accuracy_train))
        print("\tValidation Cost: {}".format(loss_valid))
        print("\tValidation Accuracy: {}".format(accuracy_valid))
        
        saver.save(sess, save_path)
        return save_path
