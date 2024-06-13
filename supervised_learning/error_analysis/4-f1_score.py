#!/usr/bin/env python3
"""
F1 Score
"""
import numpy as np


def f1_score(confusion):
    """
    Function that calculates the F1 score of a confusion matrix
    Arguments:
        - confusion is a confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels
            * classes is the number of classes
    Returns:
        A numpy.ndarray of shape (classes,) containing the F1 score
        of each class
    """
    precision = np.diag(confusion) / np.sum(confusion, axis=0)
    recall = np.diag(confusion) / np.sum(confusion, axis=1)
    return 2 * (precision * recall) / (precision + recall)
