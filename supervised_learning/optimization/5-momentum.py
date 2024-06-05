#!/usr/bin/env python3
"""
updates a variable using the gradient descent
with momentum optimization algorithm
"""
import tensorflow as tf


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
    # Compute the first moment
    v_new = beta1 * v + (1 - beta1) * grad
    # Update the variable
    var_new = var - alpha * v_new
    return var.assign(var_new), v.assign(v_new)
