def matrix_factorization_sgd_step(U, V, r, lr, reg):
    """
    Perform one SGD step for matrix factorization.
    """
    # Write code here
    e = r
    for u,v in zip(U,V):
        e -= u * v

    new_U, new_V = [], []
    for index in range(len(U)):
        ui, vi = U[index], V[index]
        new_ui = ui + lr * (e * vi - reg * ui)
        new_vi = vi + lr * (e * ui - reg * vi)
        new_U.append(new_ui)
        new_V.append(new_vi)

    return new_U, new_V