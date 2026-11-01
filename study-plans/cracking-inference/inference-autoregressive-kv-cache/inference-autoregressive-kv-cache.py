import torch

def cached_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple:
    """
    Returns (outputs, key_cache, value_cache), float32 tensors in sequence order.
    """
    cached_key= None
    cached_value = None
    results = None
    _, seq_len, dk = query.shape
    
    for t in range(0,seq_len):
        if cached_key is None:
            cached_key = key[:,t:t+1,:]
        else:
            cached_key = torch.concat((cached_key, key[:,t:t+1,:]), axis=1)
        if cached_value is None:
            cached_value = value[:,t:t+1,:]
        else:
            cached_value = torch.concat((cached_value, value[:,t:t+1,:]), axis=1)

        attention = torch.matmul(query[:,t:t+1,:], torch.transpose(cached_key,-2,-1)) / (dk ** 0.5)

        attention = torch.softmax(attention, axis=-1)
        score = torch.matmul(attention, cached_value)
        if results is None:
            results = score
        else:
            results = torch.concat((results, score), axis=1)
    return results,cached_key, cached_value

        
        
    
