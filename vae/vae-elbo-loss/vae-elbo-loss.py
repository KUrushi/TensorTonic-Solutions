import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Compute VAE ELBO loss.
    """
    # Your implementation here
    recon_loss = np.sum(np.power(x-x_recon, 2))
    kl_loss = (-0.5 * np.sum(1 + 2 * log_var - np.power(mu, 2) - np.exp(log_var)))
    total_loss = recon_loss + kl_loss
    return {
        "total":total_loss,
        "recon":recon_loss,
        "kl":kl_loss
    }