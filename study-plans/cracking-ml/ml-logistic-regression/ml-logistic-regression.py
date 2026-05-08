import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    
    X = np.asarray(X)
    y = np.asarray(y)

    w = np.zeros((X.shape[1],))
    b = 0.0
    n = X.shape[0]
    for i in range(n_iters):
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)

        w = w - lr * 1/n * np.dot(X.T, y_hat - y)
        b = b - lr * 1/n * np.sum(y_hat-y)

    return w, b
        
        
