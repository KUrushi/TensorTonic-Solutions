import torch


def grouped_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """Returns:

    output tensor of shape (batch, seq, d_model)
    """
    batch_size, seq_len, d_model = hidden_states.shape

    # --- バリデーションの追加 ---
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    if d_model % num_query_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by num_query_heads ({num_query_heads})"
        )

    d_head = d_model // num_query_heads
    group_size = num_query_heads // num_kv_heads

    # 1. 線形射影
    Q = torch.matmul(hidden_states, w_q)
    K = torch.matmul(hidden_states, w_k)
    V = torch.matmul(hidden_states, w_v)

    # 2. ヘッドへの分割と転置: (batch, heads, seq, d_head)
    Q = Q.view(batch_size, seq_len, num_query_heads, d_head).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_kv_heads, d_head).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_kv_heads, d_head).transpose(1, 2)

    # 3. GQA: K, V を group_size 回複製してヘッド数を Q に揃える
    K = torch.repeat_interleave(K, repeats=group_size, dim=1)
    V = torch.repeat_interleave(V, repeats=group_size, dim=1)

    # 4. スコア計算
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head**0.5)

    # 5. 因果マスキング
    if causal:
        mask = torch.triu(
            torch.ones(
                (seq_len, seq_len), device=scores.device, dtype=torch.bool
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

    # 6. アテンション重みと文脈ベクトルの算出
    attention_weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(attention_weights, V)

    # 7. ヘッドの結合と出力射影
    context = (
        context.transpose(1, 2)
        .contiguous()
        .view(batch_size, seq_len, d_model)
    )
    output = torch.matmul(context, w_o)
    return output