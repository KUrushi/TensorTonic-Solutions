import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here
    X = np.array(X)
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    C = np.dot(Xc.T, X) / (X.shape[0] - 1)
    eigen_value, eigen_vector = np.linalg.eig(C)
    index = np.argsort(eigen_value)[::-1]
    top_k_eigenvectors = eigen_vector[:, index[:k]]

    return np.dot(Xc, top_k_eigenvectors)
    
    