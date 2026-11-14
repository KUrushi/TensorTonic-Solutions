import jax
import jax.numpy as jnp

def hessian_vector_product(w: jax.Array, v: jax.Array, x: jax.Array, y: jax.Array, alpha: float) -> jax.Array:
    """
    Returns the float32 Hessian-vector product of shape (D,).
    """
    def _loss(w):
        return jnp.mean(jax.nn.softplus(x @ w) - y * jnp.dot(x, w)) + alpha * jnp.sum(jnp.power(w, 4)) / 4

    hessian = jax.grad(_loss)
    return jax.jvp(hessian, (w,), (v,))[1]
