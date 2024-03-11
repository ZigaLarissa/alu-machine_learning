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
    mean = np.mean(X, axis=0)
    cov = np.dot((X - mean).T, (X - mean)) / (X.shape[0])
    return mean, cov
