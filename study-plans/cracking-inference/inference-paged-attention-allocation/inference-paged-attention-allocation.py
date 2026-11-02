import torch
import math

def allocate_kv_blocks(
    seq_lengths: list,
    block_size: int,
    free_block_ids: list,
) -> tuple:
    """
    Returns (block_table, blocks_used, remaining_free_blocks), integer tensors.
    """
    max_block = 0
    blocks = []
    used_capacity = 0
    for seq_length in seq_lengths:
        block = math.ceil(seq_length / block_size)
        if block > max_block:
            max_block = block
        blocks.append(block)
        used_capacity += block
        if used_capacity  > len(free_block_ids):
            raise RuntimeError("No Capacity")


    index = 0
    kv_block_ids = []
    for block in blocks:
        kv_block = free_block_ids[index:index+block]
        while len(kv_block) < max_block:
            kv_block.append(-1)
        index += block
        kv_block_ids.append(kv_block)
    return torch.tensor(kv_block_ids, dtype=torch.int64), torch.tensor(blocks, dtype=torch.int64), torch.tensor(free_block_ids[index:], dtype=torch.int64)
        

        
    
