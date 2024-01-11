#!/usr/bin/env python3
""" counts the shape of the matrix regardless of the dimensions."""
def matrix_shape(matrix):
    shape = []

    # check if the mmatrix is a list
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else None
    return shape
