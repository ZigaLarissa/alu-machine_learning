#!/usr/bin/env python3
"""
This module contains the function to calculate the determinant of a matrix.
"""


import numpy as np


def determinant(matrix):
    """
    Calculates the determinant of a matrix.
    """
    if not isinstance(matrix, list) and not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not all(len(matrix) == len(row) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    return np.linalg.det(matrix)
