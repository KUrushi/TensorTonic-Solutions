import numpy as np

def relu(x):
    return np.where(x<0, 0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def vae_decoder(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Decode latent vectors to reconstructed data.
    """
    # Your implementation here
    hw = np.random.normal(size=(z.shape[1], output_dim))
    h = z@hw
    h = relu(h) 
    xhat_w = np.random.normal(size=(output_dim,output_dim))
    xhat = h @ xhat_w 
    xhat = sigmoid(xhat)
    return xhat
    