#!/usr/bin/env python3
'''
Calculates the Gaussian Matrix
'''


import sklearn.mixture


def gmm(X, k):
    """
    Calculates a Gaussian Mixture Model (GMM) from a dataset.

    Parameters:
    X : numpy.ndarray of shape (n, d)
        The dataset.
    k : int
        The number of clusters.

    Returns:
    pi : numpy.ndarray of shape (k,)
        The cluster priors (weights).
    m : numpy.ndarray of shape (k, d)
        The centroid means.
    S : numpy.ndarray of shape (k, d, d)
        The covariance matrices for each cluster.
    clss : numpy.ndarray of shape (n,)
        The cluster indices for each data point.
    bic : float
        The Bayesian Information Criterion (BIC) for the model.
    """
    # Fit the Gaussian Mixture Model to the data
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k)
    gmm_model.fit(X)

    # Extract the required parameters
    pi = gmm_model.weights_
    m = gmm_model.means_
    S = gmm_model.covariances_
    clss = gmm_model.predict(X)
    bic = gmm_model.bic(X)

    return pi, m, S, clss, bic
