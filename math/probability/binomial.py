#!/usr/bin/env python3
"""
binomial distribution
"""


class Binomial():
    """
    binomial distribution
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        class constructor
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not 0 < p < 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)

        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = float(sum(data) / len(data))
            summation = 0
            for i in data:
                summation += (i - mean) ** 2
            variance = (summation / len(data))
            self.p = 1 - (variance / mean)
            self.n = round(mean / self.p)
            self.p = float(mean / self.n)
