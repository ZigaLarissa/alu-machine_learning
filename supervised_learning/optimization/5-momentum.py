#!/usr/bin/env python3
"""
updates a variable using the gradient descent
with momentum optimization algorithm
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent
    with momentum optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.
        var (tf.Variable): The variable to be updated.
        grad (tf.Tensor): The gradient of var.
        v (tf.Variable): The previous first moment of var.

    Returns:
        tf.Operation: The operation that updates var.
    """
    # Momentum update
    v_new = beta1 * v + (1 - beta1) * grad

    # Variable update
    var_new = var - alpha * v_new

    return var.assign(var_new), v.assign(v_new)
