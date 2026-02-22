import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    x = torch.matmul(Q, K.transpose(1,2)) / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32))
    x = F.softmax(x, dim=2)
    x = torch.matmul(x, V)
    return x