#!/usr/bin/env python3
"""
This module contains the functions to calculate
the correlation between two variables.
"""


import numpy as np


def correlation(C):
    """
    This function calculates the correlation matrix
    for a dataset.

    Arguments:
     - C: a numpy.ndarray of shape (d, d) containing
          the covariance matrix of the data.

    Returns:
     A numpy.ndarray of shape (d, d) containing the
        correlation matrix.
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]: 
        raise ValueError("C must be a 2D square matrix")
    return np.corrcoef(C, rowvar=False)
