import torch

def per_channel_int8_quantize(
    x: torch.Tensor,
    channel_axis: int,
) -> tuple:
    """
    Returns (quantized, scale, dequantized): int8 codes, broadcastable float scales, float reconstruction.
    """
    norm_axis = channel_axis % x.dim()
    
    reduce_dims = tuple(d for d in range(x.dim()) if d != norm_axis)
    
    sc = torch.amax(torch.abs(x), dim=reduce_dims, keepdim=True)
    sc = torch.where(sc == 0.0, 1.0, sc / 127)
    qi = torch.clip(torch.round(x / sc), -127, 127).to(torch.int8)
    return (qi, sc, qi * sc)