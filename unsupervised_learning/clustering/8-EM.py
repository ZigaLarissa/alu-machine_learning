#!/usr/bin/env python3
"""
This module contains a function that performs the
Expectation Maximization (EM) algorithm for a Gaussian Mixture Model (GMM).
"""

import numpy as np
from 4_initialize import initialize
from 6_expectation import expectation
from 7_maximization import maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a Gaussian Mixture Model (GMM).

    Parameters:
    - X: numpy.ndarray of shape (n, d) containing the dataset
        - n: number of data points
        - d: number of dimensions for each data point
    - k: positive integer, the number of clusters
    - iterations: positive integer, maximum number of iterations (default is 1000)
    - tol: non-negative float, tolerance for early stopping (default is 1e-5)
    - verbose: boolean, if True, prints log likelihood every 10 iterations (default is False)

    Returns:
    - pi: numpy.ndarray of shape (k,) containing the priors for each cluster
    - m: numpy.ndarray of shape (k, d) containing the centroid means for each cluster
    - S: numpy.ndarray of shape (k, d, d) containing the covariance matrices for each cluster
    - g: numpy.ndarray of shape (k, n) containing the probabilities for each data point in each cluster
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

        # Check for convergence (difference in log likelihood)
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
