#!/usr/bin/env python3
"""
PCA
"""

import numpy as np


def pca(X, var=0.95):
    """
    Function that performs PCA on a dataset
    """
    # Normalize the mean value
    X_mean = X - np.mean(X, axis=0)

    # create the covariance matrix
    cov = np.cov(X_mean.T)

    # Eigenvalues and eigenvectors
    w, v = np.linalg.eig(cov)

    # Sort eigenvalues
    w_sort = np.sort(w)[::-1]
    w_sort = w[idx]
    v_sort = v[:, idx]

    # Explained variance
    exp_var = w_sort / np.sum(w_sort)

    # Cumulative explained variance
    cum_exp_var = np.cumsum(exp_var)

    # Number of dimensions to keep
    d = np.argwhere(cum_exp_var >= var)[0, 0]

    # Projection matrix
    W = v_sort[:, :d + 1]

    return W
