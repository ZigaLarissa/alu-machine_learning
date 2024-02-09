#!/usr/bin/env python3
"""
normal distribution
"""


class Normal():
    """
    normal distribution
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        data is a list of the data to be used to estimate the distribution
        mean is the mean of the distribution
        stddev is the standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
            self.mean = float(mean)

        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            summation = 0
            for i in data:
                summation += (i - self.mean) ** 2
            self.stddev = (summation / len(data)) ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        """
        return ((x - self.mean) / self.stddev)

    def x_value(self, z):
        """
        Calculates the x-score of a given z-value
        """
        return ((z * self.stddev) + self.mean)
