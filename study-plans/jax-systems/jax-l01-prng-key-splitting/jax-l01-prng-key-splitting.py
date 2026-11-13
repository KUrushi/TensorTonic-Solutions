import jax
import jax.numpy as jnp

def dropout_masks(seed: int, num_layers: int, shape: list[int], keep_prob: float) -> jax.Array:
    """
    Returns a bool array of shape (num_layers, followed by the entries of shape) holding one dropout keep mask per layer.
    """
    key = jax.random.key(seed)
    outputs = []
    for n in range(num_layers):
        key, subkey = jax.random.split(key)
        outputs.append(jax.random.bernoulli(subkey,keep_prob, shape=shape))

    return jnp.stack(outputs)
