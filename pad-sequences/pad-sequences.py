import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    outputs = []
    if max_len is None:
        max_len = 0
        for seq in seqs:
            max_len = max(len(seq), max_len)

    for seq in seqs:
        output = [i for i in seq[:max_len]]
        length = len(seq)
        while (length := len(output)) != max_len:
            output.append(pad_value)

        outputs.append(output)
    return np.array(outputs)
            
            
    
            