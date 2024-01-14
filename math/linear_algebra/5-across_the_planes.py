#!/usr/bin/env python3
"""
    adds two matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """
    mat1: list of lists of ints/floats
    mat2: list of lists of ints/floats
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [mat1[i][j] + mat2[i][j] for i in range(len(mat1))
            for j in range(len(mat1[0]))]
