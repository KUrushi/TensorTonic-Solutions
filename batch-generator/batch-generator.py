import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    # Write code here
    random_index = np.arange(0, len(y))
    rng = np.random.default_rng(rng)
    rng.shuffle(random_index)

    X = np.array(X)
    y = np.array(y)
    for i in range(0, len(y), batch_size):
        index = random_index[i:i+batch_size]
        x_ =X[index]
        y_ = y[index]
        if drop_last and y_.shape[0] < batch_size:
            break
        yield x_, y_
        
        
    