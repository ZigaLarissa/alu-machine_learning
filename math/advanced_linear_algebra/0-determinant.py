#!/usr/bin/env python3
"""
This module contains the function to calculate the determinant of a matrix.
"""

def smaller_matrix(matrix, i, j):
    """
    Returns the matrix with the i-th row and j-th column removed.
    """
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]

def determinant(matrix):
    """
    Calculates the determinant of a matrix.
    """
    if not isinstance(matrix, list) and not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 and len(matrix) == 1 and not all(len(matrix) == len(row) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    
    if len(matrix) == 0:
        return 1

    if len(matrix) == 1:
        return matrix[0][0]
    
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    answer = 0
    for j in range(matrix):
        answer += ((-1) ** j) * matrix[0][j] * determinant(smaller_matrix(matrix, 0, j))
    return answer
