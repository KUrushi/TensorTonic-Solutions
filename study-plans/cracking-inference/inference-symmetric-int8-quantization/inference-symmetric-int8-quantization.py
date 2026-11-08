import torch

def symmetric_int8_quantize(
    x: torch.Tensor,
) -> tuple:
    """
    Returns (quantized, scale, dequantized): int8 codes, scalar float scale, float reconstruction./
    """
    s = torch.max(torch.abs(x)) / 127

    if s == 0.0:
        s = torch.tensor(1.0)
        return x.to(dtype=torch.int8), s, x
    qi = torch.clip(torch.round(x/s),-127,127).to(torch.int8)
    x = s*qi
    return (qi, s, x)
    
