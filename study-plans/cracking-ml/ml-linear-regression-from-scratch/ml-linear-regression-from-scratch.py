import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    w = np.zeros((X.shape[1],))
    b = 0
    n = X.shape[0]
    for epoch in range(epochs):
        y_hat = np.dot(X, w) + b
        w = w - lr * 2 / n * np.dot(X.T, y_hat - y)
        b = b - lr * 2 / n * np.sum(y_hat - y)

    return np.round(w,4), np.round(b,4)
