import numpy as np

def pca(X, var=0.95):
    """
    Performs PCA on the dataset X while maintaining
     var fraction of the variance.

    Parameters:
    X : numpy.ndarray of shape (n, d)
        n is the number of data points
        d is the number of dimensions in each point
    var : float, optional (default is 0.95)
        The fraction of the variance that the PCA transformation
         should maintain

    Returns:
    W : numpy.ndarray of shape (d, nd)
        The weights matrix that maintains var fraction of X's
         original variance
    """
    # Step 1: Compute the covariance matrix
    cov_matrix = np.cov(X, rowvar=False)

    # Step 2: Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 3: Sort the eigenvalues (and corresponding eigenvectors)
    #  in descending order
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Step 4: Compute the cumulative sum of the sorted eigenvalues
    cumulative_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    # Step 5: Find the number of components needed to
    # maintain the given variance
    num_components = np.argmax(cumulative_var >= var) + 1

    # Step 6: Select the top eigenvectors corresponding
    #  to the desired number of components
    W = eigenvectors[:, :num_components]

    return W
