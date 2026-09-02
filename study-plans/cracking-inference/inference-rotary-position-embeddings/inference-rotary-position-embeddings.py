import torch
from typing import Tuple

def rotate_adjacent(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rx = torch.stack((-x2, x1), dim=-1)
    return rx.flatten(start_dim=-2)

def apply_rotary_position_embeddings(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    head_dim = query.shape[-1]
    seq_len = positions.shape[0]
    device = query.device
    dtype = query.dtype

    # 1. 周波数 theta_i の計算
    dim_indices = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    theta = 1.0 / (base ** (dim_indices / head_dim))

    # 2. 位相角 (m * theta_i) の計算と 2 要素連続配置
    m_theta = torch.outer(positions.to(dtype), theta)
    m_theta = torch.repeat_interleave(m_theta, repeats=2, dim=-1)

    cos = torch.cos(m_theta)  # (seq_len, head_dim)
    sin = torch.sin(m_theta)  # (seq_len, head_dim)

    # 3. query のテンソル形状に合わせて cos, sin のブロードキャスト形状を決定
    if query.ndim == 4:
        if query.shape[2] == seq_len:
            # (batch, num_heads, seq_len, head_dim)
            cos = cos.view(1, 1, seq_len, head_dim)
            sin = sin.view(1, 1, seq_len, head_dim)
        elif query.shape[1] == seq_len:
            # (batch, seq_len, num_heads, head_dim)
            cos = cos.view(1, seq_len, 1, head_dim)
            sin = sin.view(1, seq_len, 1, head_dim)
        else:
            raise ValueError(
                f"seq_len ({seq_len}) does not match query dimension 1 or 2: {query.shape}"
            )
    elif query.ndim == 3:
        # (batch, seq_len, head_dim)
        if query.shape[1] != seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) does not match query dimension 1: {query.shape}"
            )
        cos = cos.view(1, seq_len, head_dim)
        sin = sin.view(1, seq_len, head_dim)
    else:
        raise ValueError(f"Unsupported query dimension: {query.ndim}")

    # 4. 回転の適用
    query_rot = (query * cos) + (rotate_adjacent(query) * sin)
    key_rot = (key * cos) + (rotate_adjacent(key) * sin)

    return query_rot, key_rot