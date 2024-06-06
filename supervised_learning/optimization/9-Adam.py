#!/usr/bin/env python3
"""
updates a variable in place using the Adam optimization algorithm.
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    updates a variable in place using the Adam optimization algorithm

    args:
        alpha is the learning rate
        beta1 is the weight used for the first moment
        beta2 is the weight used for the second moment
        epsilon is a small number to avoid division by zero
        var is a numpy.ndarray containing the variable to be updated
        grad is a numpy.ndarray containing the gradient of var
        v is the previous first moment of var
        s is the previous second moment of var
        t is the time step used for bias correction

    returns:
        the updated variable, the new first moment, and the new second moment
    """
    v_corrected = v / (1 - beta1 ** t)  # Bias correction for first moment
    s_corrected = s / (1 - beta2 ** t)  # Bias correction for second moment

    var_update = alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    var -= var_update * grad  # Update the variable

    v = beta1 * v + (1 - beta1) * grad  # Update the first moment
    s = beta2 * s + (1 - beta2) * (grad ** 2)  # Update the second moment

    return var, v, s
