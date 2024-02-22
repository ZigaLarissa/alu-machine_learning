#!/usr/bin/env python3
"""
This module contains the function to calculate the determinant of a matrix.
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix.
    """
    if not isinstance(matrix, list) and not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not all(len(matrix) == len(row) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    
    if len(matrix) == 0:
        return 1

    if len(matrix) == 1:
        return matrix[0][0]
    
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    answer = 0
    for j in range(matrix):
        answer += ((-1) ** j) * matrix[0][j] * determinant([[matrix[i][j] for j in range(1, len(matrix))] for i in range(1, len(matrix))])
    return answer
