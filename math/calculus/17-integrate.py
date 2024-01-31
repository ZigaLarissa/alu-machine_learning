#!/usr/bin/env python3
"""
returns the coefficients of the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    if poly or C not valid return None
    """
    if not isinstance(poly, list) or len(poly) == 0 or not isinstance(C, int):
        return None
    elif len(poly) == 1:
        return [C]
    elif len(poly) == 1:
        return [C]
    integral_poly = [C]
    for i in range(len(poly)):
        if poly[i] % (i + 1) == 0:
            integral_poly.append(poly[i] // (i + 1))
        else:
            integral_poly.append(poly[i] / (i + 1))
    return integral_poly
