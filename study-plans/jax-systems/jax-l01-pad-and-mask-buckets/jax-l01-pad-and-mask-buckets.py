import jax
import jax.numpy as jnp

@jax.jit(static_argnames=("L",))
def _pad_kernel(tokens, lengths, pad_id, L):
    starts = jnp.cumsum(lengths) - lengths
    cols = jnp.arange(L, dtype=jnp.int32)
    mask = cols[None, :] < lengths[:, None]
    indices = starts[:, None] + cols[None, :]
    safe_indices = jnp.clip(indices, 0, tokens.shape[0]-1)
    gathered = tokens[safe_indices]
    pad_val = jnp.asarray(pad_id, dtype=jnp.int32)
    padded_tokens = jnp.where(mask, gathered, pad_val)
    return {"tokens": padded_tokens, "mask": mask}

    
    

def pad_to_bucket(tokens: jax.Array, lengths: jax.Array, buckets: list[int], pad_id: int) -> dict:
    """
    Returns a dict with tokens, an int32 JAX array of shape (B, L), and mask, a bool JAX array of shape (B, L).
    """
    max_len = jnp.max(lengths).item()
    L = min(b for b in buckets if b >= max_len)
    return _pad_kernel(tokens, lengths, pad_id, L)
    
