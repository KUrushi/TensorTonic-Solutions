import numpy as np

def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    w = np.zeros(X.shape[1])
    b = 0.0

    for epoch in range(epochs):
        for x, y_i in zip(X,y):
            z = np.dot(x, w) + b
            y_hat = np.where(z >= 0, 1.0, 0.0)
            update = lr * (y_i - y_hat)
            w += update * x
            b += update

    return w, b
    