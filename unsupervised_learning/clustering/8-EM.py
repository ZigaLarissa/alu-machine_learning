#!/usr/bin/env python3
"""
This module contains a function that does the
Expectation Maximization (EM) algorithm pour a Gaussian Mixture Model (GMM).
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Do the expectation maximization a Gaussian Mixture Model (GMM).

    Parameters:
    - X: numpy.ndarray of shape (n, d) containing the dataset
        - n: number of data points
        - d: number of dimensions each data point
    - k: positive integer, the number of clusters
    - iterations: positive integer, maximum number of iterations (default is 1000)
    - tol: non-negative float, tolerance early stopping (default is 1e-5)
    - verbose: boolean, if True, prints log likelihood every 10 iterations (default is False)

    Returns:
    - pi: numpy.ndarray of shape (k,) containing the priors each cluster
    - m: numpy.ndarray of shape (k, d) containing the centroid means each cluster
    - S: numpy.ndarray of shape (k, d, d) containing the covariance matrices each cluster
    - g: numpy.ndarray of shape (k, n) containing the probabilities each data point in each cluster
    - l: log likelihood of the model
    - None, None, None, None, None on failure
    """

    # Validate inputs
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None

    n, d = X.shape

    # Initialize parameters using the initialize function
    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    log_likelihood_old = 0

    for i in range(iterations):
        # E-step: Expectation step using the expectation function
        g, log_likelihood = expectation(X, pi, m, S)
        if g is None or log_likelihood is None:
            return None, None, None, None, None

        # M-step: Maximization step using the maximization function
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        # Check convergence (difference in log likelihood)
        if abs(log_likelihood - log_likelihood_old) <= tol:
            break

        log_likelihood_old = log_likelihood

        # Print log likelihood every 10 iterations and after the last iteration if verbose is True
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {log_likelihood:.5f}")

    # Final log likelihood output
    if verbose:
        print(f"Log Likelihood after {i + 1} iterations: {log_likelihood:.5f}")

    # Return the final parameters
    return pi, m, S, g, log_likelihood
