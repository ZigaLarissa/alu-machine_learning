#!/usr/bin/env python3
"""
This module contains the function to calculate the definiteness of a matrix.
"""
import numpy as np


def is_positive_definite(matrix):
    """
    defines if a matrix is positive definite
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.shape[0] != matrix.shape[1]:
        return False
    if not np.all(np.linalg.eigvals(matrix) > 0):
        return False
    return True


def is_positive_semidefinite(matrix):
    """
    defines if a matrix is positive semidefinite
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2:
        return False
    if matrix.shape[0] != matrix.shape[1]:
        return False
    if not np.all(np.linalg.eigvals(matrix) >= 0):
        return False
    return True


def is_negative_definite(matrix):
    """
    defines if a matrix is negative definite
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.shape[0] != matrix.shape[1]:
        return False
    if not np.all(np.linalg.eigvals(matrix) < 0):
        return False
    return True


def is_negative_semidefinite(matrix):
    """
    defines if a matrix is negative semidefinite
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.shape[0] != matrix.shape[1]:
        return False
    if not np.all(np.linalg.eigvals(matrix) <= 0):
        return False
    return True


def is_indefinite(matrix):
    """
    defines if a matrix is indefinite
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.shape[0] != matrix.shape[1]:
        return False
    if not any(np.linalg.eigvals(matrix) > 0) and not any(np.linalg.eigvals(matrix) < 0):
        return False
    return True
