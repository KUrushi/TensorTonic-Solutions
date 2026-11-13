import jax
import jax.numpy as jnp

def weighted_histogram(counts: jax.Array, indices: jax.Array, weights: jax.Array) -> jax.Array:
    """
    Returns a new float32 array of shape (K,) holding counts plus every weight added into the bin named by its index, with out-of-range indices ignored.
    """
    weights = jnp.where(indices >= 0, weights, 0.0)
    return counts.at[indices].add(weights)