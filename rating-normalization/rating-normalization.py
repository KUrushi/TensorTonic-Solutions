def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    # Write code here
    outputs = []
    num_of_users = len(matrix)
    for row in matrix:
        counts = sum(1 for _ in row if _ != 0)
        if counts == 0:
            outputs.append([0.0 for _ in row])
            continue
            
        mean = sum(row) / counts
        output_row = [float(r - mean) if r != 0 else 0.0 for r in row]
        outputs.append(output_row)
    return outputs
    