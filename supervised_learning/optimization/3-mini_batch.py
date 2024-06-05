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
    # Start a TensorFlow session
    with tf.Session() as sess:
        # Load the model
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        # Retrieve the necessary tensors and operations from the graph
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        y = graph.get_tensor_by_name('y:0')
        accuracy = graph.get_tensor_by_name('accuracy:0')
        loss = graph.get_tensor_by_name('loss:0')
        train_op = graph.get_operation_by_name('train_op')

        m = X_train.shape[0]  # Number of training examples
        step_number = 0

        # Loop over epochs
        for epoch in range(epochs):
            # Shuffle the training data
            X_train, Y_train = shuffle_data(X_train, Y_train)
            
            # Loop over batches
            for i in range(0, m, batch_size):
                end = i + batch_size
                X_batch = X_train[i:end]
                Y_batch = Y_train[i:end]
                
                # Train the model on the current batch
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                
                # Print step metrics every 100 steps
                if step_number % 100 == 0:
                    step_cost, step_accuracy = sess.run([loss, accuracy], feed_dict={x: X_batch, y: Y_batch})
                    print(f"\tStep {step_number}:")
                    print(f"\t\tCost: {step_cost}")
                    print(f"\t\tAccuracy: {step_accuracy}")
                
                step_number += 1
            
            # After each epoch, calculate and print metrics for the entire training and validation sets
            train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            
            print(f"After {epoch + 1} epochs:")
            print(f"\tTraining Cost: {train_cost}")
            print(f"\tTraining Accuracy: {train_accuracy}")
            print(f"\tValidation Cost: {valid_cost}")
            print(f"\tValidation Accuracy: {valid_accuracy}")
        
        # Save the trained model
        saver.save(sess, save_path)
    
    return save_path
