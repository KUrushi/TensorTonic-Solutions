import torch

def multi_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns: output tensor of shape (batch, seq, d_model)
    """
    batch_size, seq_len, d_model = hidden_states.shape
    d_head = d_model // num_query_heads

    Q = torch.matmul(hidden_states, w_q)
    K = torch.matmul(hidden_states, w_k)
    V = torch.matmul(hidden_states, w_v)

    Q = Q.view(batch_size, seq_len, num_query_heads, d_head).transpose(1,2)
    K = K.view(batch_size, seq_len, 1, d_head).transpose(1,2)
    V = V.view(batch_size, seq_len, 1, d_head).transpose(1,2)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5)

    if causal:
        mask = torch.triu(
            torch.ones((seq_len, seq_len), device=scores.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    attention_weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(attention_weights, V)
    context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    output = torch.matmul(context, w_o)
    return output


    
