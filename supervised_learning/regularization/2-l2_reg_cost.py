#!/usr/bin/env python3


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.
    
    Parameters:
    cost -- tensor containing the cost of the network without L2 regularization
    lambtha -- L2 regularization parameter
    weights -- dictionary of the weights of the neural network
    m -- number of data points
    
    Returns:
    The cost of the network accounting for L2 regularization
    """
    l2_term = 0
    for key in weights:
        if key.startswith('W'):
            l2_term += np.sum(np.square(weights[key]))
    l2_cost = cost + (lambtha / (2 * m)) * l2_term
    return l2_cost
