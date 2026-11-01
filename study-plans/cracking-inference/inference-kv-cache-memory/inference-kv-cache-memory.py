import torch

def kv_cache_memory_bytes(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_query_heads: int,
    gqa_kv_heads: int,
    head_dim: int,
    mla_latent_dim: int,
    mla_rotary_key_dim: int,
    bytes_per_element: int,
) -> torch.Tensor:
    """
    Returns an int64 tensor of byte counts ordered [MHA, MQA, GQA, MLA].
    """
    basic_c = batch_size * seq_len * num_layers * bytes_per_element
    mha_size = basic_c * 2 * num_query_heads * head_dim
    mqa_size = basic_c * 2 * head_dim
    gqa_size = basic_c * 2 * head_dim * gqa_kv_heads
    mla_size = basic_c * (mla_latent_dim + mla_rotary_key_dim)
    return torch.tensor([mha_size, mqa_size, gqa_size, mla_size], dtype=torch.int64)
    