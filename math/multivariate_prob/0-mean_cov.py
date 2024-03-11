#!/usr/bin/env python3
import numpy as np
"""
This module contains the function to calculate the mean and covariance of a multivariate distribution.
"""


def mean_cov(X):
    """
    This function calculates the mean and covariance of
    a multivariate distribution.
    """
    if type(X) is not np.ndarray:
        raise TypeError("X must be a 2D numpy.ndarray")
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0)
    cov = np.dot((X - mean).T, (X - mean)) / (X.shape[0] - 1)
    return mean, cov
