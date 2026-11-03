import torch

def match_prefix_cache(
    request_token_ids: list,
    cached_token_blocks: list,
    cached_physical_block_ids: list,
    block_size: int,
) -> tuple:
    """
    Returns (matched_token_count, reusable_physical_block_ids), scalar and 1D integer tensors.
    """
    request_token_ids = torch.tensor(request_token_ids, dtype=torch.int64)
    # 4で割り切れる最大サイズにスライスして端数を切り捨てる
    truncated_len = (len(request_token_ids) // block_size) * block_size
    v_truncated = request_token_ids[:truncated_len].reshape(-1, block_size)

    # 4列の行列に変換
    matrix = v_truncated.reshape(-1, block_size)
    
    max_hit = torch.tensor(0)
    max_physical_block = torch.tensor([],dtype=torch.int64)
    for cached_token_block, cached_physical_block_id in zip(
        cached_token_blocks, cached_physical_block_ids
    ):
        cached_token_block = torch.tensor(cached_token_block, dtype=torch.int64)
        min_block_size = min(cached_token_block.shape[0],matrix.shape[0]) 
        cached_token_block, cached_matrix =  cached_token_block[:min_block_size, :], matrix[:min_block_size, :]
        cache_hits = (cached_token_block == cached_matrix).all(dim=1)
        running_hits = cache_hits.to(torch.int64).cumprod(dim=0)
        num_hits = running_hits.sum()

        if num_hits > max_hit:
            max_hit = num_hits
            max_physical_block = torch.tensor(cached_physical_block_id[:num_hits], dtype=torch.int64)

    return max_hit*block_size, max_physical_block
       
