import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    num_samples = len(seqs)
    if max_len is None:
        max_len = max([len(s) for s in seqs], default=0)

    dtype = np.int32
    out = np.full((num_samples, max_len), pad_value, dtype=dtype)
    
    for i, seq in enumerate(seqs):
        actual_length = min(max_len, len(seq))
        out[i, :actual_length] = seq[:actual_length]

    return out

            
            
    
            