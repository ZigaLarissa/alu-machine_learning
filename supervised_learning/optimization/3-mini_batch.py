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
