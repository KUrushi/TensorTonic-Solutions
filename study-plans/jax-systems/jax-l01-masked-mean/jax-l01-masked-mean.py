import jax
import jax.numpy as jnp

def masked_mean(x: jax.Array, mask: jax.Array, axis: int) -> jax.Array:
    """
    Returns a float32 array with the shape of x without the given axis, holding the mean of the valid entries.
    """
    masked_x = jnp.where(mask, x, 0.0)
    masked_sum = jnp.sum(masked_x, axis=axis)
    V = jnp.maximum(mask.sum(axis=axis),1)
    return masked_sum / V
