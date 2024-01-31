#!/usr/bin/env python3
"""
calculates the sum of the first n**2 natural numbers
"""


def summation_i_squared(n):
    sum_of_squares = (n * (n + 1) * ((2 * n) + 1)) // 6
    return sum_of_squares
