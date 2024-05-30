#!/usr/bin/env python3
"""
 Updates the weights of a neural network with
 Dropout regularization using gradient descent.
"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network
    """
    m = Y.shape[1]
    weights_c = weights.copy()
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache["A" + str(i - 1)]
        A_prev = cache["A" + str(i - 1)]
        W = weights_c["W" + str(i)]
        b = weights_c["b" + str(i)]
        dW = (1 / m) * np.matmul(dz, A_prev.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        weights["W" + str(i)] = weights_c["W" + str(i)] - alpha * dW
        weights["b" + str(i)] = weights_c["b" + str(i)] - alpha * db
        if i - 1 > 0:
            dz = np.matmul(W.T, dz) * (1 - (A * A))
            if i - 1 in cache.keys():
                if i - 1 in cache.keys():
                    D = np.random.rand(A_prev.shape[0], A_prev.shape[1])
                    D = np.where(D < keep_prob, 1, 0)
                    A = A * D / keep_prob
                    cache["A" + str(i - 1)] = A
    return weights
