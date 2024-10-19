#!/usr/bin/env python3
'''
Calculates K means clustering on a dataset
'''


import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means clustering on a dataset.

    Parameters:
    X : numpy.ndarray of shape (n, d)
        The dataset.
    k : int
        The number of clusters.

    Returns:
    C : numpy.ndarray of shape (k, d)
        The centroid means for each cluster.
    clss : numpy.ndarray of shape (n,)
        The index of the cluster in C that each data point belongs to.
    """
    # Perform K-means clustering
    kmeans_model = sklearn.cluster.KMeans(n_clusters=k)
    kmeans_model.fit(X)

    # Extract the centroid means
    C = kmeans_model.cluster_centers_

    # Get the index of the cluster for each data point
    clss = kmeans_model.labels_

    return C, clss
