import numpy as np

def vae_encoder(x: np.ndarray, latent_dim: int) -> tuple:
    """
    Encode input to latent distribution parameters.
    """
    # Your implementation here
    rng = np.random.default_rng(0)
    h_w = rng.normal(size=(x.shape[1], latent_dim))
    h = x @ h_w
    u_w = rng.normal(size=(latent_dim, latent_dim))
    sigma_w = rng.normal(size=(latent_dim, latent_dim))
    mu = h @ u_w
    sigma = h @ u_w
    return mu,sigma