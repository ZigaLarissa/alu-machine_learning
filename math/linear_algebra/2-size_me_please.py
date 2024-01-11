#!/usr/bin/env python3
# def the function
def matrix_shape(matrix):
    shape = []
    # check if the mmatrix is a list
    while isinstance(matrix, list):
        # if a list append its length to shape
        shape.append(len(matrix))
        # change the matrix to the first index of what was matrix before
        matrix = matrix[0] if matrix else None
    return shape
