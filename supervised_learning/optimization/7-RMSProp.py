#!/usr/bin/env python3
"""
updates a variable using the RMSProp optimization algorithm.
"""
import tensorflow as tf


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    updates a variable using the RMSProp optimization algorithm.

    args:
        alpha: the learning rate
        beta2: the RMSProp weight
        epsilon: a small number to avoid division by zero
        var: a np.ndarray containing the variable to be updated
        grad: a np.ndarray containing the gradient of var
        s: the previous second moment of var

    returns:
        the updated variable and the new moment
    """
    s = beta2 * s + (1 - beta2) * grad ** 2
    var = var - alpha * grad / (s ** 0.5 + epsilon)
    return var, s
