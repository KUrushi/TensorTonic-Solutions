import math

def cyclic_encoding(values, period):
    """
    Encode cyclic features as sin/cos pairs.
    """
    # Write code here
    outputs = []
    for value in values:
        theta = 2 * math.pi * value / period
        outputs.append([math.sin(theta),math.cos(theta)])
    return outputs