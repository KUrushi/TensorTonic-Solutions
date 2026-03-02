def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    results = []
    for i in range(len(X)):
        y_i = []
        for j in range(len(b)):
            y_ij = 0
            for k in range(len(W)):
                y_ij += X[i][k] * W[k][j] 
            y_ij += b[j]
            y_i.append(y_ij)

        results.append(y_i)
    return results
