import jax
import jax.numpy as jnp

def _loss(params, x, y, l2):
    
    w = jnp.asarray(params['w'])
    b = params['b']

    zn = x @ w + b
    return jnp.mean(jax.nn.softplus(zn) - y * zn) + l2 * jnp.sum(jnp.power(w,2)) / 2

def logistic_value_and_grad(params: dict, x: jax.Array, y: jax.Array, l2: float) -> dict:
    """
    Returns a dict with the float32 scalar loss and a grads dict with keys w and b.
    """
    loss, grads =  jax.value_and_grad(_loss)(params, x, y, l2)
    return {"loss": loss, "grads": grads}
    
    
    
    
