import torch
from typing import Tuple

def multi_head_latent_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_down: torch.Tensor,
    w_up_k: torch.Tensor,
    w_up_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (output tensor of shape (batch, seq, d_model), latent tensor of shape (batch, seq, d_latent))
    """
    c = torch.matmul(hidden_states, w_down)
    K = torch.matmul(c, w_up_k)
    V = torch.matmul(c, w_up_v)
    Q = torch.matmul(hidden_states, w_q)

    batch_size, seq_len, d_model = hidden_states.shape
    d_head = d_model // num_heads
    Q = Q.view(batch_size, seq_len, num_heads, d_head).transpose(1,2)
    K = K.view(batch_size, seq_len, num_heads, d_head).transpose(1,2)
    V = V.view(batch_size, seq_len, num_heads, d_head).transpose(1,2)

    scores = torch.matmul(Q, K.transpose(-1,-2)) / (d_head ** 0.5)
    if causal:
        mask = torch.triu(
            torch.ones((seq_len, seq_len), device=scores.device, dtype=torch.bool),
            diagonal=1
        )

        scores = scores.masked_fill(mask, float("-inf"))

    attention_weight = torch.softmax(scores, dim=-1)
    context = torch.matmul(attention_weight, V)

    context = context.transpose(1,2).contiguous().view(batch_size, seq_len, d_model)
    output = torch.matmul(context, w_o)
    return (output, c)
