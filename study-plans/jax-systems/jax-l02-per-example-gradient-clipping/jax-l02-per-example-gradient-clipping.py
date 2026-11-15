import jax
import jax.numpy as jnp

def per_example_clipped_gradient(params: dict, x: jax.Array, y: jax.Array, max_norm: float) -> dict:
    """
    Returns a dict with the averaged clipped grads (keys w and b) and the float32 per-example norms.
    """

    def _loss(params, x_i, y_i):
        pred = x_i @ params['w'] + params['b']
        return jnp.power(pred - y_i, 2) / 2

    def _per_example_gradient_clipping(params, x_i, y_i):
        grad = jax.grad(_loss)(params, x_i, y_i)
        leaves = jax.tree.leaves(grad)
        norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in leaves))
        scale = jnp.maximum(1.0, norm /max_norm)
        clipped_grad = jax.tree.map(lambda g: g / scale, grad)
        return clipped_grad, norm
    clipped_grads, per_example_norms = jax.vmap(
        _per_example_gradient_clipping, in_axes=(None, 0, 0)
    )(params, x, y)
    clipped_grads = jax.tree_util.tree_map(lambda g : jnp.mean(g, axis=0), clipped_grads)
    return {"grads":clipped_grads, "norms":per_example_norms}
    
        
